import dataclasses
import json
import logging
import os
import sys
import time

import aim
import beartype
import chex
import datasets
import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import torch
import torchvision.transforms.v2 as transforms
import tyro
from jaxtyping import Array, Float, Int, jaxtyped

import frx

IMAGENET_CHANNEL_MEAN = (0.4632, 0.4800, 0.3762)
IMAGENET_CHANNEL_STD = (0.2375, 0.2291, 0.2474)

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger("train")


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Args:
    seed: int = 42

    # Data
    resize_size: int = 256
    """how big to resize images."""
    crop_size: int = 224
    """after resize, how big an image to crop."""
    n_classes: int = 1_000
    """number of classes (1000 for ImageNet-1K)."""
    v2_dir: str = "."
    """Where ImageNet-V2 is stored."""
    batch_size: int = 256
    """train and evaluation batch size."""
    n_workers: int = 4
    """number of dataloader workers"""
    p_mixup: float = 0.2
    """probability of adding MixUp to a batch."""
    pin_memory: bool = False
    """whether to pin memory in the dataloader."""

    # Optimization
    learning_rate: float = 0.001
    beta1: float = 0.9
    beta2: float = 0.999
    grad_clip: float = 1.0
    grad_accum: int = 1
    """number of steps to accumulate gradients for. `1` implies no accumulation."""
    weight_decay: float = 0.0001
    n_warmup_steps: int = 10_000
    n_epochs: int = 90
    """number of epochs to train for."""

    # Logging
    log_every: int = 10
    """how often to log metrics."""
    track: bool = True
    """whether to track with Aim."""
    ckpt_dir: str = os.path.join(".", "checkpoints")
    """where to store model checkpoints."""


class DataloaderMixup:
    def __init__(self, args: Args):
        self.p_mixup = args.p_mixup
        self.n_classes = args.n_classes
        self.mixup = transforms.MixUp(num_classes=args.n_classes)

    def __call__(self, batch: list[object]) -> dict[str, object]:
        batch = torch.utils.data.default_collate(batch)
        if torch.rand(()) < self.p_mixup:
            batch["image"], batch["label"] = self.mixup(batch["image"], batch["label"])
        else:
            batch["label"] = torch.nn.functional.one_hot(
                batch["label"], num_classes=self.n_classes
            ).to(torch.float32)
        return batch


@beartype.beartype
def make_dataloader(args: Args, dataset, *, is_train: bool):
    drop_last = is_train
    shuffle = is_train

    # Transforms
    transform = []
    transform.append(transforms.Resize(args.resize_size, antialias=True))
    if is_train:
        train_transforms = [
            transforms.RandomResizedCrop(
                args.crop_size,
                scale=(0.08, 1.0),
                ratio=(0.75, 4.0 / 3.0),
                antialias=True,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(2, 10),
        ]
        transform.extend(train_transforms)
    else:
        transform.append(transforms.CenterCrop(args.crop_size))

    common_transforms = [
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(mean=IMAGENET_CHANNEL_MEAN, std=IMAGENET_CHANNEL_STD),
    ]
    transform.extend(common_transforms)
    transform = transforms.Compose(transform)

    if shuffle:
        dataset.shuffle(args.seed)

    def hf_transform(example):
        example["image"] = example["image"].convert("RGB")
        example["image"] = transform(example["image"])
        return example

    dataset = dataset.to_iterable_dataset().map(hf_transform).with_format("torch")

    collate_fn = (
        DataloaderMixup(args)
        if is_train and args.p_mixup > 0
        else torch.utils.data.default_collate
    )

    return torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        drop_last=drop_last,
        num_workers=min(args.n_workers, dataset.n_shards),
        pin_memory=args.pin_memory,
        persistent_workers=min(args.n_workers, dataset.n_shards) > 0,
        shuffle=False,  # We use dataset.shuffle instead
        collate_fn=collate_fn,
    )


def save(filename, cfg, model):
    with open(filename, "wb") as fd:
        cfg_str = json.dumps(cfg)
        fd.write((cfg_str + "\n").encode("utf-8"))
        eqx.tree_serialise_leaves(fd, model)


@jaxtyped(typechecker=beartype.beartype)
@eqx.filter_value_and_grad
def compute_grads(
    model: eqx.Module,
    images: Float[Array, "batch 3 width height"],
    labels: Float[Array, "batch n_class"],
    *,
    keys: list[chex.PRNGKey],
):
    logits = jax.vmap(model, in_axes=(0, None, 0))(images, False, jnp.array(keys))
    loss = optax.safe_softmax_cross_entropy(logits, labels)

    return jnp.mean(loss)


@jaxtyped(typechecker=beartype.beartype)
@eqx.filter_jit
def step_model(
    model: eqx.Module,
    optim: optax.GradientTransformation | optax.MultiSteps,
    state: optax.OptState | optax.MultiStepsState,
    images: Float[Array, "batch 3 width height"],
    labels: Float[Array, "batch n_class"],
    *,
    keys: list[chex.PRNGKey],
):
    loss, grads = compute_grads(model, images, labels, keys=keys)
    updates, new_state = optim.update(grads, state, model)

    model = eqx.apply_updates(model, updates)

    return model, new_state, loss


@jaxtyped(typechecker=beartype.beartype)
def evaluate(model: eqx.Module, dataloader, key: chex.PRNGKey) -> dict[str, object]:
    """ """

    @jaxtyped(typechecker=beartype.beartype)
    @eqx.filter_jit
    def _compute_loss(
        model: eqx.Module,
        images: Float[Array, "b 3 w h"],
        keys,
        labels: Int[Array, " b"],
    ) -> tuple[Float[Array, ""], Float[Array, "b n_classes"]]:
        logits = jax.vmap(model, in_axes=(0, None, 0))(images, True, jnp.array(subkeys))
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
        return loss.item(), logits

    metrics = {"loss": []}

    for batch in dataloader:
        images = jnp.asarray(batch["image"])
        labels = jnp.asarray(batch["label"])
        key, *subkeys = jax.random.split(key, num=len(labels) + 1)
        loss, logits = _compute_loss(model, images, jnp.array(subkeys), labels)
        metrics["loss"].append(loss)

        _, indices = jax.lax.top_k(logits, k=5)

        for k in (1, 5):
            _, indices = jax.lax.top_k(logits, k=k)
            n_correct = jnp.any(indices == labels[:, None], axis=1).sum()

            name = f"acc{k}"
            if name not in metrics:
                metrics[name] = []
            metrics[name].append(n_correct / len(labels))
    metrics = {key: jnp.mean(jnp.array(value)).item() for key, value in metrics.items()}
    return metrics


@beartype.beartype
def main(args: Args):
    key = jax.random.key(seed=args.seed)
    key, model_key = jax.random.split(key)

    # 1. Model
    model_cfg = dict(
        d=384,
        hidden_d=1_536,
        n_heads=6,
        n_layers=12,
        p_dropout=0.2,
        patch_size=16,
        n_patches=196,
        n_classes=args.n_classes,
    )
    model = frx.VisionTransformer(**model_cfg, key=model_key)

    # 2. Dataset
    dataset = datasets.load_dataset(
        "ILSVRC/imagenet-1k", split="train", trust_remote_code=True
    ).train_test_split(test_size=0.01, shuffle=True, seed=args.seed)
    datasets.disable_progress_bars()
    train_dataset = dataset.pop("train")
    minival_dataset = dataset.pop("test")

    val_dataset = datasets.load_dataset(
        "ILSVRC/imagenet-1k", split="validation", trust_remote_code=True
    )
    v2_dataset = datasets.load_dataset(
        "imagefolder", data_dir=args.v2_dir, split="train"
    )

    train_dataloader = make_dataloader(args, train_dataset, is_train=True)
    val_dataloaders = {
        "minival": make_dataloader(args, minival_dataset, is_train=False),
        "val": make_dataloader(args, val_dataset, is_train=False),
        "v2": make_dataloader(args, v2_dataset, is_train=False),
    }

    # 3. Train
    n_steps_per_epoch = int(len(train_dataset) / args.batch_size / args.grad_accum)
    n_steps = n_steps_per_epoch * args.n_epochs
    lr_schedule = optax.schedules.warmup_cosine_decay_schedule(
        0.0, args.learning_rate, args.n_warmup_steps, n_steps
    )
    optim = optax.adamw(
        learning_rate=lr_schedule,
        b1=args.beta1,
        b2=args.beta2,
        weight_decay=args.weight_decay,
    )
    if args.grad_clip > 0:
        optim = optax.chain(optim, optax.clip_by_global_norm(args.grad_clip))
    if args.grad_accum > 1:
        optim = optax.MultiSteps(optim, every_k_schedule=args.grad_accum)

    state = optim.init(eqx.filter(model, eqx.is_inexact_array))

    # 4. Logging and checkpointing
    if args.track:
        run = aim.Run(experiment="train")
        run["hparams"] = {k: frx.to_aim_value(v) for k, v in vars(args).items()}
        run["hparams"]["cmd"] = " ".join([sys.executable] + sys.argv)

    os.makedirs(args.ckpt_dir, exist_ok=True)

    logger.info("Training for %d steps.", n_steps)

    flops_per_iter = 0
    flops_promised = 38.7e12  # 38.7 TFLOPS for fp16 on A6000
    global_step = 0
    start_time = time.time()

    t1 = time.time()
    for epoch in range(args.n_epochs):
        for b, batch in enumerate(train_dataloader):
            t0 = t1
            t1 = time.time()
            key, *subkeys = jax.random.split(key, num=args.batch_size + 1)

            images = jnp.asarray(batch["image"])
            labels = jnp.asarray(batch["label"])

            model, state, loss = step_model(
                model, optim, state, images, labels, keys=subkeys
            )
            global_step += 1

            if global_step % args.log_every == 0:
                step_per_sec = global_step / (time.time() - start_time)
                dt = t1 - t0
                metrics = {
                    "train_loss": loss.item(),
                    "step_per_sec": step_per_sec,
                    "learning_rate": lr_schedule(global_step // args.grad_accum).item(),
                    "mfu": flops_per_iter / dt / flops_promised,
                }
                if args.track:
                    run.track(metrics, step=global_step)
                logger.info(
                    "step: %d, loss: %.5f, step/sec: %.1f",
                    global_step,
                    loss.item(),
                    step_per_sec,
                )

            if global_step == 10:
                # Calculate flops one time after a couple iterations.
                logger.info("Calculating FLOPs per forward/backward pass.")
                flops_per_iter = (
                    eqx.filter_jit(step_model)
                    .lower(model, optim, state, images, labels, keys=subkeys)
                    .compile()
                    .compiled.cost_analysis()[0]["flops"]
                )
                logger.info("Calculated FLOPs: %d.", flops_per_iter)

        # 4. Evaluate
        # We want to evaluate on the rest of the training set (minival) as well as (1) the true validation set (2) imagenet v2 and (3) imagenet real. Luckily this is the same 1K classes so we can simply do inference without any fitting.
        for name, dataloader in val_dataloaders.items():
            key, subkey = jax.random.split(key)
            logger.info("Evaluating %s.", name)
            metrics = evaluate(model, dataloader, subkey)
            metrics = {f"{name}_{key}": value for key, value in metrics.items()}
            if args.track:
                run.track(metrics, step=global_step)
            logger.info(
                ", ".join(f"{key}: {value:.3f}" for key, value in metrics.items()),
            )

        # Checkpoint.
        save(os.path.join(args.ckpt_dir, f"{run.hash}_ep{epoch}.eqx"), model_cfg, model)


if __name__ == "__main__":
    main(tyro.cli(Args))

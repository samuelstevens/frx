import dataclasses
import json
import logging
import os
import sys
import time
import typing

import beartype
import chex
import datasets
import equinox as eqx
import jax
import jax.experimental.mesh_utils as mesh_utils
import jax.numpy as jnp
import optax
import torch
import torchvision.transforms.v2 as transforms
import tyro
from jaxtyping import Array, Float, Int, jaxtyped

import wandb

from . import helpers, mup, tracking, vit

Schedule = typing.Literal[None, "warmup", "warmup+cosine-decay"]


log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger(__name__)


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Args:
    seed: int = 42
    """Random seed."""

    # Model
    p_dropout: float = 0.2
    """dropout probability."""
    model_d: int = 128
    """hidden dimension of ViT."""
    n_layers: int = 6
    """number of transformer layers."""
    init_std: float = 0.02
    """std dev of normal distribution for weight initialization."""

    # Data
    resize_size: int = 256
    """How big to resize images."""
    crop_size: int = 224
    """After resize, how big an image to crop."""
    n_classes: int = 1_000
    """Number of classes (1000 for ImageNet-1K)."""
    v2_dir: str = ""
    """Where ImageNet-V2 is stored."""
    batch_size: int = 256
    """Train and evaluation batch size."""
    n_workers: int = 4
    """Number of dataloader workers"""
    p_mixup: float = 0.2
    """Probability of adding MixUp to a batch."""
    pin_memory: bool = False
    """Whether to pin memory in the dataloader."""

    # Optimization
    learning_rate: float = 0.001
    """Peak learning rate."""
    lr_schedule: Schedule = "warmup"
    """What kind of learning rate schedule to use."""
    n_lr_warmup: int = 10_000
    """Number of learning rate warmup steps."""
    beta1: float = 0.9
    """Adam beta1."""
    beta2: float = 0.999
    """Adam beta2."""
    grad_clip: float = 1.0
    """Maximum gradient norm. `0` implies no clipping."""
    grad_accum: int = 1
    """Number of steps to accumulate gradients for. `1` implies no accumulation."""
    weight_decay: float = 0.0001
    """Weight decay applied to Optax's AdamW optimizer."""
    n_epochs: int = 90
    """Number of epochs to train for."""

    # muP
    do_mup: bool = True
    """Apply muP init, learning rate scaling, and QK head scaling."""
    mup_base_d: int = 128
    """what the original d was."""

    # Logging
    log_every: int = 10
    """how often to log metrics."""
    track: bool = True
    """whether to track with Aim."""
    ckpt_dir: str = os.path.join(".", "checkpoints")
    """where to store model checkpoints."""
    tags: list[str] = dataclasses.field(default_factory=list)
    """any tags for this specific run."""


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
        transforms.Normalize(
            mean=helpers.IMAGENET_CHANNEL_MEAN, std=helpers.IMAGENET_CHANNEL_STD
        ),
    ]
    transform.extend(common_transforms)
    transform = transforms.Compose(transform)

    if shuffle:
        dataset.shuffle(args.seed)

    def hf_transform(example):
        example["image"] = example["image"].convert("RGB")
        example["image"] = transform(example["image"])
        return example

    dataset = (
        dataset.to_iterable_dataset(num_shards=args.n_workers)
        .map(hf_transform)
        .with_format("torch")
    )

    collate_fn = (
        DataloaderMixup(args)
        if is_train and args.p_mixup > 0
        else torch.utils.data.default_collate
    )

    return torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        drop_last=drop_last,
        num_workers=args.n_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.n_workers > 0 and is_train,
        shuffle=False,  # We use dataset.shuffle instead
        collate_fn=collate_fn,
    )


@beartype.beartype
def get_lr_schedule(
    args: Args, n_train: int, *, lr: float | None = None
) -> optax.Schedule:
    # Total number of training steps.
    n_steps_per_epoch = int(n_train / args.batch_size)
    n_steps = n_steps_per_epoch * args.n_epochs

    logger.info("Training for %d steps.", n_steps)

    if lr is None:
        lr = args.learning_rate

    # No schedule
    if not args.lr_schedule:
        return optax.constant_schedule(lr)

    # Linear warmup, constant LR after.
    if args.lr_schedule == "warmup":
        return optax.schedules.warmup_constant_schedule(0.0, lr, args.n_lr_warmup)

    # Linear warmup + cosine decay (fixed number of training steps).
    if args.lr_schedule == "warmup+cosine-decay":
        return optax.schedules.warmup_cosine_decay_schedule(
            0.0, lr, args.n_lr_warmup, n_steps
        )

    typing.assert_never(args.lr_schedule)


def save(filename, cfg, model):
    with open(filename, "wb") as fd:
        cfg_str = json.dumps(cfg)
        fd.write((cfg_str + "\n").encode("utf-8"))
        eqx.tree_serialise_leaves(fd, model)


@jaxtyped(typechecker=beartype.beartype)
def compute_loss(
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
@eqx.filter_jit(donate="all")
def step_model(
    model: eqx.Module,
    optim: optax.GradientTransformation | optax.MultiSteps,
    state: optax.OptState | optax.MultiStepsState,
    images: Float[Array, "batch 3 width height"],
    labels: Float[Array, "batch n_class"],
    *,
    keys: list[chex.PRNGKey],
):
    loss, grads = eqx.filter_value_and_grad(compute_loss)(
        model, images, labels, keys=keys
    )
    (updates,), new_state = optim.update([grads], state, [model])

    model = eqx.apply_updates(model, updates)

    return model, new_state, loss


@jaxtyped(typechecker=beartype.beartype)
def evaluate(model: eqx.Module, dataloader, key: chex.PRNGKey) -> dict[str, object]:
    """ """

    @jaxtyped(typechecker=beartype.beartype)
    @eqx.filter_jit(donate="all-except-first")
    def _compute_loss(
        model: eqx.Module,
        images: Float[Array, "b 3 w h"],
        keys,
        labels: Int[Array, " b"],
    ) -> tuple[Float[Array, ""], Float[Array, "b n_classes"]]:
        logits = jax.vmap(model, in_axes=(0, None, 0))(images, True, jnp.array(subkeys))
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
        return loss, logits

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
def train(args: Args) -> str:
    key = jax.random.key(seed=args.seed)
    key, model_key = jax.random.split(key)

    # 1. Model
    model_cfg = dict(
        d=args.model_d,
        hidden_d=args.model_d * 4,
        n_heads=16,
        n_layers=args.n_layers,
        p_dropout=args.p_dropout,
        patch_size=16,
        n_patches=196,
        n_classes=args.n_classes,
    )
    if args.do_mup:
        model = vit.VisionTransformerMuP(**model_cfg, key=model_key)
    else:
        model = vit.VisionTransformer(**model_cfg, key=model_key)

    if args.do_mup:
        key, model_key = jax.random.split(key)
        model = mup.init(
            model, std=args.init_std, m_d=args.model_d / args.mup_base_d, key=model_key
        )
    logger.info("Initialized model.")

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

    train_dataloader = make_dataloader(args, train_dataset, is_train=True)
    val_dataloaders = {
        "minival": make_dataloader(args, minival_dataset, is_train=False),
        "val": make_dataloader(args, val_dataset, is_train=False),
    }
    if args.v2_dir:
        v2_dataset = datasets.load_dataset(
            "imagefolder", data_dir=args.v2_dir, split="train"
        )
        val_dataloaders["v2"] = make_dataloader(args, v2_dataset, is_train=False)
    logger.info("Loaded %d dataloaders.", len(val_dataloaders) + 1)

    # 3. Train
    lr_schedule = get_lr_schedule(args, len(train_dataset))
    optim = optax.adamw(
        learning_rate=lr_schedule,
        b1=args.beta1,
        b2=args.beta2,
        weight_decay=args.weight_decay,
    )

    if args.do_mup:
        param_labels = jax.tree.map(
            lambda _: "hidden", eqx.filter(model, eqx.is_inexact_array)
        )
        param_labels = eqx.tree_at(
            lambda m: m.patch_embedding.linear.weight, param_labels, "embedding"
        )
        param_labels = eqx.tree_at(
            lambda m: m.patch_embedding.linear.bias, param_labels, "embedding"
        )
        param_labels = eqx.tree_at(lambda m: m.pos_embedding, param_labels, "embedding")

        # Make a new optimizer for hidden (non-embedding) params that scales learning rate by 1/m_d.
        scaled_lr = args.learning_rate / (args.model_d / args.mup_base_d)
        hidden_adamw = optax.adamw(
            # Note that we specify lr so we can override args.learning_rate with the scaled LR.
            learning_rate=get_lr_schedule(args, len(train_dataset), lr=scaled_lr),
            b1=args.beta1,
            b2=args.beta2,
            weight_decay=args.weight_decay,
        )
        optim = optax.multi_transform(
            {"hidden": hidden_adamw, "embedding": optim}, [param_labels]
        )

    if args.grad_clip > 0:
        if args.do_mup:
            logger.warning("Gradient clipping with muP likely doesn't work.")
        optim = optax.chain(optim, optax.clip_by_global_norm(args.grad_clip))
    if args.grad_accum > 1:
        optim = optax.MultiSteps(optim, every_k_schedule=args.grad_accum)

    state = optim.init(eqx.filter([model], eqx.is_inexact_array))
    logger.info("Initialized optimizer.")

    # 4. Multi-device training
    n_devices = len(jax.local_devices())
    logger.info("Training on %d devices.", n_devices)

    if n_devices > 1 and n_devices % 2 != 0:
        logger.warning(
            "There are %d devices, which is an odd number for multi-GPU training.",
            n_devices,
        )
    # Image batches have four dimensions: batch x channels x width x height. We want to
    # split the batch dimension up over all devices. The same applies to labels, but
    # they only have batch x classes
    image_sharding = jax.sharding.PositionalSharding(
        mesh_utils.create_device_mesh((n_devices, 1, 1, 1))
    )
    label_sharding = jax.sharding.PositionalSharding(
        mesh_utils.create_device_mesh((n_devices, 1))
    )
    # We replicate() the sharding because we want an exact copy of the model and
    # optimizer state on each device.
    model, state = eqx.filter_shard((model, state), image_sharding.replicate())

    # 5. Logging and checkpointing
    mode = "online" if args.track else "disabled"
    hparams = {k: helpers.to_primitive(v) for k, v in vars(args).items()}
    hparams["cmd"] = " ".join([sys.executable] + sys.argv)
    run = wandb.init(
        project="frx",
        entity="samuelstevens",
        config=hparams,
        tags=args.tags,
        mode=mode,
        reinit=True,
    )

    os.makedirs(args.ckpt_dir, exist_ok=True)

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

            images = eqx.filter_shard(jnp.asarray(batch["image"]), image_sharding)
            labels = eqx.filter_shard(jnp.asarray(batch["label"]), label_sharding)

            model, state, loss = step_model(
                model, optim, state, images, labels, keys=subkeys
            )
            global_step += 1

            if global_step % args.log_every == 0:
                step_per_sec = global_step / (time.time() - start_time)
                dt = t1 - t0
                metrics = {
                    "train/loss": loss.item(),
                    "schedule/learning_rate": lr_schedule(
                        global_step // args.grad_accum
                    ).item(),
                    "perf/step_per_sec": step_per_sec,
                    "perf/mfu": flops_per_iter / dt / flops_promised,
                }
                run.log(metrics, step=global_step)
                logger.info(
                    "step: %d, loss: %.5f, step/sec: %.2f",
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
            metrics = {f"{name}/{key}": value for key, value in metrics.items()}
            run.log(metrics, step=global_step)
            logger.info(
                ", ".join(f"{key}: {value:.3f}" for key, value in metrics.items()),
            )
        # Record epoch at this step only once.
        run.log({"epoch": epoch}, step=global_step)

        # Checkpoint.
        save(os.path.join(args.ckpt_dir, f"{run.id}_ep{epoch}.eqx"), model_cfg, model)

    # At the end of the run:
    # 1. Save to sqlite storage.
    # 2. Finish WandB run.
    # 3. Return WandB run id.
    tracking.save(run.id, hparams, dict(run.summary))
    run.finish()
    return run.id


if __name__ == "__main__":
    train(tyro.cli(Args))

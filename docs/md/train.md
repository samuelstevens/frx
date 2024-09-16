Module train
============

Functions
---------

`compute_grads(model: equinox._module.Module, images: jaxtyping.Float[Array, 'batch 3 width height'], labels: jaxtyping.Float[Array, 'batch n_class'], *, keys: list[jax.Array])`
:   

`evaluate(model: equinox._module.Module, dataloader, key: jax.Array) ‑> dict[str, object]`
:   

`main(args: train.Args)`
:   

`make_dataloader(args: train.Args, dataset, *, is_train: bool)`
:   

`save(filename, cfg, model)`
:   

`step_model(model: equinox._module.Module, optim: optax._src.base.GradientTransformation | optax.transforms._accumulation.MultiSteps, state: Union[jax.Array, numpy.ndarray, numpy.bool, numpy.number, Iterable[ForwardRef('ArrayTree')], Mapping[Any, ForwardRef('ArrayTree')], optax.transforms._accumulation.MultiStepsState], images: jaxtyping.Float[Array, 'batch 3 width height'], labels: jaxtyping.Float[Array, 'batch n_class'], *, keys: list[jax.Array])`
:   

Classes
-------

`Args(seed: int = 42, resize_size: int = 256, crop_size: int = 224, n_classes: int = 1000, v2_dir: str = '.', batch_size: int = 256, n_workers: int = 4, p_mixup: float = 0.2, pin_memory: bool = False, learning_rate: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, grad_clip: float = 1.0, grad_accum: int = 1, weight_decay: float = 0.0001, n_warmup_steps: int = 10000, n_epochs: int = 90, log_every: int = 10, track: bool = True, ckpt_dir: str = './checkpoints')`
:   Args(seed: int = 42, resize_size: int = 256, crop_size: int = 224, n_classes: int = 1000, v2_dir: str = '.', batch_size: int = 256, n_workers: int = 4, p_mixup: float = 0.2, pin_memory: bool = False, learning_rate: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, grad_clip: float = 1.0, grad_accum: int = 1, weight_decay: float = 0.0001, n_warmup_steps: int = 10000, n_epochs: int = 90, log_every: int = 10, track: bool = True, ckpt_dir: str = './checkpoints')

    ### Class variables

    `batch_size: int`
    :   train and evaluation batch size.

    `beta1: float`
    :

    `beta2: float`
    :

    `ckpt_dir: str`
    :   where to store model checkpoints.

    `crop_size: int`
    :   after resize, how big an image to crop.

    `grad_accum: int`
    :   number of steps to accumulate gradients for. `1` implies no accumulation.

    `grad_clip: float`
    :

    `learning_rate: float`
    :

    `log_every: int`
    :   how often to log metrics.

    `n_classes: int`
    :   number of classes (1000 for ImageNet-1K).

    `n_epochs: int`
    :   number of epochs to train for.

    `n_warmup_steps: int`
    :

    `n_workers: int`
    :   number of dataloader workers

    `p_mixup: float`
    :   probability of adding MixUp to a batch.

    `pin_memory: bool`
    :   whether to pin memory in the dataloader.

    `resize_size: int`
    :   how big to resize images.

    `seed: int`
    :

    `track: bool`
    :   whether to track with Aim.

    `v2_dir: str`
    :   Where ImageNet-V2 is stored.

    `weight_decay: float`
    :

`DataloaderMixup(args: train.Args)`
:
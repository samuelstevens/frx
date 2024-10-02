"""
Launch script to kick off a hyperparameter sweep using a Slurm cluster (with submitit).
"""

import collections.abc
import dataclasses
import logging
import os.path
import tomllib
import typing

import beartype
import jax
import jax.numpy as jnp
import submitit
import tyro

import frx.train

Primitive = float | int | bool | str

Distribution = typing.TypedDict(
    "Distribution", {"min": float, "max": float, "dist": typing.Literal["loguniform"]}
)

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger("launch")


@beartype.beartype
def main(
    config_file: str,
    /,
    n_per_discrete: int = 1,
    override: frx.train.Args = frx.train.Args(),
    slurm: bool = False,
    n_cpus: int = 0,
    n_gpus: int = 0,
    n_hours: int = 0,
    sacct: str = "",
):
    """
    Start a hyperparameter sweep of training runs using either a Slurm cluster or a local GPU. Results are written to a sqlite file, which can be queried for final metrics to make plots like those you see in SAE papers (comparing sparsity and reconstruction loss).

    Args:
        configs: list of config filepaths.
        n_per_discrete: number of random samples to draw for each *discrete* config.
        override: individual arguments that you want to override for all jobs.
        slurm: whether to use a slurm cluster for running jobs or a local GPU.
        n_cpus: (slurm only) how many cpus to use; should be at least as many as `frx.train.Args.n_workers`.
        n_gpus: (slurm only) how many gpus to use.
        n_hours: (slurm only) how many hours to run a slurm job for.
        sacct: (slurm only) the slurm account.
    """
    with open(config_file, "rb") as fd:
        sweep_config = tomllib.load(fd)

    configs = list(expand(sweep_config, n_per_discrete=n_per_discrete))
    if len(configs) <= 1:
        msg = f"There is only one (1) concrete experiment in your config. This isn't very useful. Try adding lists to '{config_file}' or making --n-per-discrete bigger than 1."
        raise ValueError(msg)

    logger.info("Sweep has %d experiments.", len(configs))

    sweep_args, errs = [], []
    for i, config in enumerate(configs):
        try:
            args = frx.train.Args(**config)
            # Want to apply a different seed to each config.
            args = dataclasses.replace(args, seed=i)
            sweep_args.append(args)
        except Exception as err:
            errs.append(str(err))

    if errs:
        msg = "\n\n".join(errs)
        raise RuntimeError(msg)

    if slurm:
        if not n_gpus or not n_cpus or not n_hours or not sacct:
            msg = "You must specify --n-gpus, --n-cpus, --n-hours and --sacct when using --slurm."
            raise ValueError(msg)

        executor = submitit.SlurmExecutor(folder="logs")
        executor.update_parameters(
            time=n_hours * 60,
            gpus_per_node=n_gpus,
            cpus_per_task=n_cpus,
            stderr_to_stdout=True,
            partition="gpu",
            account=sacct,
            # For whatever reason, we cannot import jax without a GPU. If you set JAX_PLATFORMS=cpu to run this launcher script, then it will be true for the submitted jobs. This means that your training jobs will run on the CPU instead of the cluster GPUs. This extra arg exports an updated JAX_PLATFORMS variable for the cluster jobs.
            setup=["export JAX_PLATFORMS=''"],
        )
    else:
        executor = submitit.DebugExecutor(folder="logs")

    # Include filename in experiment tags.
    exp_name, _ = os.path.splitext(os.path.basename(config_file))
    sweep_args = [
        dataclasses.replace(
            overwrite(args, override), tags=args.tags + [exp_name], seed=args.seed + i
        )
        for i, args in enumerate(sweep_args)
    ]
    jobs = executor.map_array(frx.train.train, sweep_args)
    for i, result in enumerate(submitit.helpers.as_completed(jobs)):
        exp_id = result.result()
        logger.info("Finished task %s (%d/%d)", exp_id, i + 1, len(jobs))


@beartype.beartype
def overwrite(args: frx.train.Args, override: frx.train.Args) -> frx.train.Args:
    """
    If there are any non-default values in override, returns a copy of `args` with all those values included.

    Arguments:
        args: sweep args
        override: incoming args with zero or more non-default values.

    Returns:
        frx.train.Args
    """
    override_dict = {
        field.name: getattr(override, field.name)
        for field in dataclasses.fields(override)
        if getattr(override, field.name) != field.default
    }
    return dataclasses.replace(args, **override_dict)


@beartype.beartype
def expand_discrete(
    config: dict[str, Primitive | list[Primitive] | Distribution],
) -> collections.abc.Iterator[dict[str, Primitive]]:
    """
    Expands any list values in `config`.
    """
    if not config:
        yield config
        return

    key, value = config.popitem()

    if isinstance(value, list):
        # Expand
        for c in expand_discrete(config):
            for v in value:
                yield {**c, key: v}
    else:
        for c in expand_discrete(config):
            yield {**c, key: value}


@beartype.beartype
def expand(
    config: dict[str, Primitive | list[Primitive] | Distribution],
    *,
    n_per_discrete: int,
) -> collections.abc.Iterator[dict[str, Primitive]]:
    discrete_configs = list(expand_discrete(config))
    for config in discrete_configs:
        yield from sample_from(config, n=n_per_discrete)


@beartype.beartype
def sample_from(
    config: dict[str, Primitive | Distribution], *, n: int
) -> collections.abc.Iterator[dict[str, Primitive]]:
    # 1. Count the number of distributions and collect random fields
    random_fields = {k: v for k, v in config.items() if isinstance(v, dict)}
    dim = len(random_fields)

    # 2. Sample for each distribution
    values = roberts_sequence(n, dim, perturb=True, key=jax.random.key(seed=0))

    # 3. Scale each sample based on the min/max/dist
    scaled_values = {}
    for (key, dist), column in zip(random_fields.items(), values.T):
        if dist["dist"] == "loguniform":
            scaled = jnp.exp(
                jnp.log(dist["min"])
                + column * (jnp.log(dist["max"]) - jnp.log(dist["min"]))
            )
        elif dist["dist"] == "uniform":
            scaled = dist["min"] + column * (dist["max"] - dist["min"])
        else:
            typing.assert_never(dist["dist"])

        scaled_values[key] = scaled

    # 4. Return the sampled configs
    for i in range(n):
        yield {
            **{k: v for k, v in config.items() if not isinstance(v, dict)},
            **{k: v[i].item() for k, v in scaled_values.items()},
        }


def _newton_raphson(f, x, iters):
    """Use the Newton-Raphson method to find a root of the given function."""

    def update(x, _):
        y = x - f(x) / jax.grad(f)(x)
        return y, None

    x, _ = jax.lax.scan(update, 1.0, length=iters)
    return x


def roberts_sequence(
    num_points: int,
    dim: int,
    root_iters: int = 10_000,
    complement_basis: bool = True,
    perturb: bool = True,
    key: jax.typing.ArrayLike | None = None,
    dtype=float,
):
    """
    Returns the Roberts sequence, a low-discrepancy quasi-random sequence:
    Low-discrepancy sequences are useful for quasi-Monte Carlo methods.
    Reference:
    Martin Roberts. The Unreasonable Effectiveness of Quasirandom Sequences.
    extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences
    Args:
      num_points: Number of points to return.
      dim: The dimensionality of each point in the sequence.
      root_iters: Number of iterations to use to find the root.
      complement_basis: Complement the basis to improve precision, as described
        in https://www.martysmods.com/a-better-r2-sequence.
      key: a PRNG key.
      dtype: optional, a float dtype for the returned values (default float64 if
        jax_enable_x64 is true, otherwise float32).
    Returns:
      An array of shape (num_points, dim) containing the sequence.

    From https://github.com/jax-ml/jax/pull/23808
    """

    def f(x):
        return x ** (dim + 1) - x - 1

    root = _newton_raphson(f, jnp.astype(1.0, dtype), root_iters)

    basis = 1 / root ** (1 + jnp.arange(dim, dtype=dtype))

    if complement_basis:
        basis = 1 - basis

    n = jnp.arange(num_points, dtype=dtype)
    x = n[:, None] * basis[None, :]

    if perturb:
        if key is None:
            raise ValueError("key cannot be None when perturb=True")
        key, subkey = jax.random.split(key)
        x += jax.random.uniform(subkey, [dim])

    x, _ = jnp.modf(x)

    return x


if __name__ == "__main__":
    tyro.cli(main)

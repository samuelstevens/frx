import marimo

__generated_with = "0.8.15"
app = marimo.App(width="full")


@app.cell
def __():
    import marimo as mo

    return (mo,)


@app.cell
def __():
    import os

    # Set this to True to run the model on CPU only.
    USE_CPU_ONLY = True

    flags = os.environ.get("XLA_FLAGS", "")
    flags += " --xla_force_host_platform_device_count=8"  # Simulate 8 devices
    # Enforce CPU-only execution
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["XLA_FLAGS"] = flags
    os.environ["JAX_PLATFORMS"] = "cpu"
    return USE_CPU_ONLY, flags, os


@app.cell
def __():
    import beartype
    import equinox as eqx

    # import matplotlib.pyplot as plt
    import jax
    import jax.numpy as jnp
    import numpy as np
    import optax
    from jax.experimental import mesh_utils
    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P
    from jax.sharding import PositionalSharding
    from jaxtyping import Array, Float, jaxtyped

    import frx

    return (
        Array,
        Float,
        Mesh,
        NamedSharding,
        P,
        PositionalSharding,
        beartype,
        eqx,
        frx,
        jax,
        jaxtyped,
        jnp,
        mesh_utils,
        np,
        optax,
    )


@app.cell
def __(chex, jax):
    def fold_rng_over_axis(rng: chex.PRNGKey, axis_name: str) -> chex.PRNGKey:
        """Folds the random number generator over the given axis.

        This is useful for generating a different random number for each device
        across a certain axis (e.g. the model axis).

        Args:
            rng: The random number generator.
            axis_name: The axis name to fold the random number generator over.

        Returns:
            A new random number generator, different for each device index along the axis.
        """
        axis_index = jax.lax.axis_index(axis_name)
        return jax.random.fold_in(rng, axis_index)

    return (fold_rng_over_axis,)


@app.cell
def __(frx, jax):
    model_cfg = dict(
        d=32,
        hidden_d=128,
        n_heads=1,
        n_layers=2,
        p_dropout=0,
        patch_size=4,
        n_patches=64,
        n_classes=10,
    )
    model = frx.VisionTransformer(**model_cfg, key=jax.random.key(seed=0))
    return model, model_cfg


@app.cell
def __(jax):
    batch_size = 128

    images = jax.random.normal(jax.random.key(seed=1), (batch_size, 3, 32, 32))
    labels = jax.random.randint(jax.random.key(seed=2), (batch_size,), 0, 10)
    return batch_size, images, labels


@app.cell
def __(jax, mesh_utils):
    n_devices = len(jax.local_devices())
    in_sharding = jax.sharding.PositionalSharding(
        mesh_utils.create_device_mesh((n_devices, 1, 1, 1))
    )
    out_sharding = jax.sharding.PositionalSharding(
        mesh_utils.create_device_mesh((n_devices,))
    )
    return in_sharding, n_devices, out_sharding


@app.cell
def __(eqx, in_sharding, model):
    eqx.filter_shard(model, in_sharding.replicate())
    return


@app.cell
def __(eqx, images, in_sharding, jax):
    jax.debug.visualize_array_sharding(
        eqx.filter_shard(images, in_sharding)[:, 0, :, 0]
    )
    return


@app.cell
def __(eqx, jax, labels, out_sharding):
    jax.debug.visualize_array_sharding(eqx.filter_shard(labels, out_sharding))
    return


@app.cell
def __(in_sharding):
    in_sharding
    return


@app.cell
def __(in_sharding):
    in_sharding.replicate()
    return


@app.cell
def __(batch_size, eqx, images, in_sharding, jax, jnp, model):
    keys = jax.random.split(jax.random.key(seed=0), batch_size)
    logits = jax.vmap(
        eqx.filter_shard(model, in_sharding.replicate()), in_axes=(0, None, 0)
    )(eqx.filter_shard(images, in_sharding), False, jnp.array(keys))
    return keys, logits


@app.cell
def __(jax, logits):
    jax.debug.visualize_array_sharding(logits)
    return


@app.cell
def __(eqx, jax, labels, logits, optax, out_sharding):
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits, eqx.filter_shard(labels, out_sharding)
    )
    jax.debug.visualize_array_sharding(loss)
    return (loss,)


@app.cell
def __(jnp, loss):
    jnp.mean(loss).sharding
    return


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()

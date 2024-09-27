import chex
import equinox as eqx
import jax
import jax.numpy as jnp


def init(model: eqx.Module, *, std: float, m_d: float, key: chex.PRNGKey) -> eqx.Module:
    """
    Re-init all eqx.nn.Linear modules to have weights sampled from ~N(0, std^2 / m_d).
    The Eluether [post](https://blog.eleuther.ai/mutransfer/) specifies not changing the embedding weight initialization.
    But that's for language.
    For vision, where our inputs are already dense, we want the patch embedding to also be reduced by m_d.
    """
    is_linear = lambda x: isinstance(x, eqx.nn.Linear)  # noqa: E731
    get_weights = lambda m: [  # noqa: E731
        x.weight
        for x in jax.tree_util.tree_leaves(m, is_leaf=is_linear)
        if is_linear(x)
    ]
    weights = get_weights(model)
    new_weights = [
        jax.random.normal(subkey, weight.shape, weight.dtype) * std / jnp.sqrt(m_d)
        for weight, subkey in zip(weights, jax.random.split(key, len(weights)))
    ]
    return eqx.tree_at(get_weights, model, new_weights)

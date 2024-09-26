import beartype
import chex
import einops
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, jaxtyped


@jaxtyped(typechecker=beartype.beartype)
class PatchEmbedding(eqx.Module):
    linear: eqx.nn.Linear
    patch_size: int

    def __init__(
        self, input_ch: int, output_d: int, patch_size: int, *, key: chex.PRNGKey
    ):
        self.patch_size = patch_size

        self.linear = eqx.nn.Linear(
            self.patch_size * self.patch_size * input_ch, output_d, key=key
        )

    def __call__(
        self, x: Float[Array, "channels width height"]
    ) -> Float[Array, "n_patches d"]:
        x = einops.rearrange(
            x,
            "c (w pw) (h ph) -> (w h) (c pw ph)",
            ph=self.patch_size,
            pw=self.patch_size,
        )
        x = jax.vmap(self.linear)(x)

        return x


@jaxtyped(typechecker=beartype.beartype)
class AttentionBlock(eqx.Module):
    layer_norm1: eqx.nn.LayerNorm
    layer_norm2: eqx.nn.LayerNorm
    attn: eqx.nn.MultiheadAttention
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear
    dropout1: eqx.nn.Dropout
    dropout2: eqx.nn.Dropout

    def __init__(
        self,
        input_d: int,
        hidden_d: int,
        n_heads: int,
        p_dropout: float,
        *,
        key: chex.PRNGKey,
    ):
        attn_key, key1, key2 = jax.random.split(key, 3)
        self.layer_norm1 = eqx.nn.LayerNorm(input_d)
        self.layer_norm2 = eqx.nn.LayerNorm(input_d)
        self.attn = eqx.nn.MultiheadAttention(n_heads, input_d, key=attn_key)

        self.linear1 = eqx.nn.Linear(input_d, hidden_d, key=key1)
        self.linear2 = eqx.nn.Linear(hidden_d, input_d, key=key2)

        self.dropout1 = eqx.nn.Dropout(p_dropout)
        self.dropout2 = eqx.nn.Dropout(p_dropout)

    def __call__(
        self, x: Float[Array, "n_patches d"], inference: bool, key: chex.PRNGKey
    ) -> Float[Array, "n_patches d"]:
        key1, key2 = jax.random.split(key)

        x_ = jax.vmap(self.layer_norm1)(x)
        x = x + self.attn(x_, x_, x_)

        x_ = jax.vmap(self.layer_norm2)(x)
        x_ = jax.vmap(self.linear1)(x_)
        x_ = jax.nn.gelu(x_)

        x_ = self.dropout1(x_, inference=inference, key=key1)
        x_ = jax.vmap(self.linear2)(x_)
        x_ = self.dropout2(x_, inference=inference, key=key2)

        x = x + x_
        return x


@jaxtyped(typechecker=beartype.beartype)
class VisionTransformer(eqx.Module):
    patch_embedding: PatchEmbedding
    pos_embedding: Float[Array, "..."]
    attn_blocks: list[AttentionBlock]
    dropout: eqx.nn.Dropout
    head: eqx.nn.Linear
    n_layers: int

    def __init__(
        self,
        d: int,
        hidden_d: int,
        n_heads: int,
        n_layers: int,
        p_dropout: float,
        patch_size: int,
        n_patches: int,
        n_classes: int,
        *,
        key: chex.PRNGKey,
    ):
        key1, key2, key3, key4 = jax.random.split(key, 4)
        self.patch_embedding = PatchEmbedding(3, d, patch_size, key=key1)
        self.pos_embedding = jax.random.normal(key2, (n_patches + 1, d))
        self.dropout = eqx.nn.Dropout(p_dropout)

        self.n_layers = n_layers
        self.attn_blocks = [
            AttentionBlock(d, hidden_d, n_heads, p_dropout, key=key_)
            for key_ in jax.random.split(key3, self.n_layers)
        ]
        self.head = eqx.nn.Linear(d, n_classes, key=key4)

    def __call__(
        self, x: Float[Array, "3 width height"], inference: bool, key: chex.PRNGKey
    ) -> Float[Array, " n_classes"]:
        x = self.patch_embedding(x)
        x += self.pos_embedding[: x.shape[0]]
        dropout_key, *attn_keys = jax.random.split(key, self.n_layers)
        x = self.dropout(x, inference=inference, key=dropout_key)
        for block, attn_key in zip(self.attn_blocks, attn_keys):
            x = block(x, inference, key=attn_key)
        x = jnp.mean(x, axis=0)
        x = self.head(x)
        return x

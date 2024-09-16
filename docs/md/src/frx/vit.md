Module src.frx.vit
==================

Classes
-------

`AttentionBlock(input_d: int, hidden_d: int, n_heads: int, p_dropout: float, *, key: jax.Array)`
:   AttentionBlock(input_d: int, hidden_d: int, n_heads: int, p_dropout: float, *, key: jax.Array)

    ### Ancestors (in MRO)

    * equinox._module.Module

    ### Class variables

    `attn: equinox.nn._attention.MultiheadAttention`
    :

    `dropout1: equinox.nn._dropout.Dropout`
    :

    `dropout2: equinox.nn._dropout.Dropout`
    :

    `layer_norm1: equinox.nn._normalisation.LayerNorm`
    :

    `layer_norm2: equinox.nn._normalisation.LayerNorm`
    :

    `linear1: equinox.nn._linear.Linear`
    :

    `linear2: equinox.nn._linear.Linear`
    :

`PatchEmbedding(input_ch: int, output_d: int, patch_size: int, *, key: jax.Array)`
:   PatchEmbedding(input_ch: int, output_d: int, patch_size: int, *, key: jax.Array)

    ### Ancestors (in MRO)

    * equinox._module.Module

    ### Class variables

    `linear: equinox.nn._linear.Linear`
    :

    `patch_size: int`
    :

`VisionTransformer(d: int, hidden_d: int, n_heads: int, n_layers: int, p_dropout: float, patch_size: int, n_patches: int, n_classes: int, *, key: jax.Array)`
:   VisionTransformer(d: int, hidden_d: int, n_heads: int, n_layers: int, p_dropout: float, patch_size: int, n_patches: int, n_classes: int, *, key: jax.Array)

    ### Ancestors (in MRO)

    * equinox._module.Module

    ### Class variables

    `attn_blocks: list[src.frx.vit.AttentionBlock]`
    :

    `dropout: equinox.nn._dropout.Dropout`
    :

    `head: equinox.nn._linear.Linear`
    :

    `n_layers: int`
    :

    `patch_embedding: src.frx.vit.PatchEmbedding`
    :

    `pos_embedding: jaxtyping.Float[Array, '...']`
    :
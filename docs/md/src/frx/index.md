Module src.frx
==============

Sub-modules
-----------
* src.frx.helpers
* src.frx.vit

Functions
---------

`to_aim_value(value: object)`
:   Recursively converts objects into [Aim](https://github.com/aimhubio/aim)-compatible values.
    
    As a fallback, tries to call `to_aim_value()` on an object.

Classes
-------

`DummyAimRun(*args, **kwargs)`
:   

    ### Methods

    `track(self, metrics: dict[str, object], step: int)`
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
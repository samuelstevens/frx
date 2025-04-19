import marimo

__generated_with = "0.8.15"
app = marimo.App(width="medium")


@app.cell
def __():
    import math

    import altair as alt
    import jax.numpy as jnp
    import marimo as mo
    import matplotlib.pyplot as plt
    import polars as pl

    return alt, jnp, math, mo, pl, plt


@app.cell
def __(mo):
    mo.md(
        """
        > The coordinate check test is a simple and cheap way to test your implementation and should be your first verification step.
        >
        > As we explained in the previous section, the goal of $\mu$P is to ensure the magnitude of the distribution of all activations is independent of any change in model width. To achieve this, activations must be controlled at initialization and after every training step. The coordinate check test involves training models of different widths for 10 steps. During each training step, we record the average size of activations for each layer type.
        >
        > We train a two layer GPT-2 model for ten steps for several different widths and five seeds.
        >
        > [Source](https://blog.eleuther.ai/mutransfer/)

        To do this, we will:

        1. Write Jax code to record mean activation size of our ViT.
        2. Train for 10 steps.
        3. Plot the sizes over time (10 steps) for different widths and depths.
        4. Convince ourselves that standard parameterization (SP) does not work.
        """
    )
    return


@app.cell
def __():
    import functools

    import beartype
    import chex
    import datasets
    import equinox as eqx
    import jax
    import optax
    import torch
    import torchvision.transforms.v2 as transforms
    from jaxtyping import Array, Float, jaxtyped

    import frx

    return (
        Array,
        Float,
        beartype,
        chex,
        datasets,
        eqx,
        frx,
        functools,
        jax,
        jaxtyped,
        optax,
        torch,
        transforms,
    )


@app.cell
def __(Array, Float, beartype, chex, eqx, frx, jax, jaxtyped, jnp, math):
    @jaxtyped(typechecker=beartype.beartype)
    class MuParamAttentionBlock(frx.vit.AttentionBlock):
        def __call__(
            self, x: Float[Array, "n_patches d"], inference: bool, key: chex.PRNGKey
        ) -> tuple[
            Float[Array, "n_patches d"], tuple[Float[Array, "..."], Float[Array, "..."]]
        ]:
            key1, key2 = jax.random.split(key)

            def process_heads(
                query_heads: Float[Array, "seq_length num_heads qk_size"],
                key_heads: Float[Array, "seq_length num_heads qk_size"],
                value_heads: Float[Array, "seq_length num_heads vo_size"],
            ) -> tuple[
                Float[Array, "seq_length num_heads qk_size"],
                Float[Array, "seq_length num_heads qk_size"],
                Float[Array, "seq_length num_heads vo_size"],
            ]:
                query_heads = query_heads / math.sqrt(query_heads.shape[-1])

                return query_heads, key_heads, value_heads

            x_ = jax.vmap(self.layer_norm1)(x)
            a = self.attn(x_, x_, x_, process_heads=process_heads)
            x = x + a

            x_ = jax.vmap(self.layer_norm2)(x)
            x_ = jax.vmap(self.linear1)(x_)
            x_ = jax.nn.gelu(x_)

            x_ = self.dropout1(x_, inference=inference, key=key1)
            x_ = jax.vmap(self.linear2)(x_)
            f = x_
            x_ = self.dropout2(x_, inference=inference, key=key2)

            x = x + x_
            return x, (a, f)

    class ViTActivations(eqx.Module):
        patch: Float[Array, "..."]
        attn: Float[Array, "..."]
        ffn: Float[Array, "..."]
        logits: Float[Array, "..."]

    class MuParamVisionTransformer(frx.VisionTransformer):
        patch_embedding: frx.vit.PatchEmbedding
        pos_embedding: Float[Array, "..."]
        attn_blocks: list[MuParamAttentionBlock]
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
            self.patch_embedding = frx.vit.PatchEmbedding(3, d, patch_size, key=key1)
            self.pos_embedding = jax.random.normal(key2, (n_patches + 1, d))
            self.dropout = eqx.nn.Dropout(p_dropout)

            self.n_layers = n_layers
            self.attn_blocks = [
                MuParamAttentionBlock(d, hidden_d, n_heads, p_dropout, key=key_)
                for key_ in jax.random.split(key3, self.n_layers)
            ]
            self.head = eqx.nn.Linear(d, n_classes, key=key4)

        def __call__(
            self, x: Float[Array, "3 width height"], inference: bool, key: chex.PRNGKey
        ) -> Float[Array, " n_classes"]:
            x = self.patch_embedding(x)
            x += self.pos_embedding[: x.shape[0]]
            patch = x

            dropout_key, *attn_keys = jax.random.split(key, self.n_layers + 1)
            x = self.dropout(x, inference=inference, key=dropout_key)

            attn_acts, ffn_acts = [], []
            for block, attn_key in zip(self.attn_blocks, attn_keys):
                x, (a, f) = block(x, inference, key=attn_key)
                attn_acts.append(a)
                ffn_acts.append(f)
            x = jnp.mean(x, axis=0)
            x = self.head(x)

            return x, ViTActivations(
                patch, jnp.array(attn_acts), jnp.array(ffn_acts), x
            )

    return MuParamAttentionBlock, MuParamVisionTransformer, ViTActivations


@app.cell
def __(beartype, datasets, frx, torch, transforms):
    batch_size = 64
    learning_rate = 1e-2
    n_seeds = 3
    n_steps = 5

    @beartype.beartype
    def make_dataloader(dataset, *, n_workers: int = 4):
        dataset.shuffle(42)

        # Transforms
        transform = [
            transforms.Resize(256, antialias=True),
            transforms.RandomResizedCrop(
                224, scale=(0.08, 1.0), ratio=(0.75, 4.0 / 3.0), antialias=True
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(2, 10),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(
                mean=frx.IMAGENET_CHANNEL_MEAN, std=frx.IMAGENET_CHANNEL_STD
            ),
        ]
        transform = transforms.Compose(transform)

        def hf_transform(example):
            example["image"] = example["image"].convert("RGB")
            example["image"] = transform(example["image"])
            return example

        dataset = (
            dataset.to_iterable_dataset(num_shards=n_workers)
            .map(hf_transform)
            .with_format("torch")
        )

        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            drop_last=True,
            num_workers=n_workers,
            pin_memory=False,
            persistent_workers=True,
            shuffle=False,  # We use dataset.shuffle instead
        )

    train_dataset = (
        datasets.load_dataset(
            "ILSVRC/imagenet-1k", split="train", trust_remote_code=True
        )
        .train_test_split(test_size=0.01, shuffle=True, seed=42)
        .pop("train")
    )

    train_dataloader = make_dataloader(train_dataset)
    return (
        batch_size,
        learning_rate,
        make_dataloader,
        n_seeds,
        n_steps,
        train_dataloader,
        train_dataset,
    )


@app.cell
def __(Array, Float, Int, beartype, chex, eqx, jax, jaxtyped, jnp, optax):
    @jaxtyped(typechecker=beartype.beartype)
    def compute_loss(
        model: eqx.Module,
        images: Float[Array, "batch 3 width height"],
        labels: Int[Array, " batch"],
        *,
        keys: list[chex.PRNGKey],
    ) -> tuple[Float[Array, ""], Float[Array, "..."]]:
        logits, acts = jax.vmap(model[0], in_axes=(0, None, 0))(
            images, False, jnp.array(keys)
        )
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)

        return jnp.mean(loss), acts

    @jaxtyped(typechecker=beartype.beartype)
    def step_model(
        model: eqx.Module,
        optim: optax.GradientTransformation | optax.MultiSteps,
        state: optax.OptState | optax.MultiStepsState,
        images: Float[Array, "batch 3 width height"],
        labels: Int[Array, " batch"],
        *,
        keys: list[chex.PRNGKey],
    ):
        (loss, acts), grads = eqx.filter_value_and_grad(compute_loss, has_aux=True)(
            model, images, labels, keys=keys
        )
        updates, new_state = optim.update(grads, state, model)

        model = eqx.apply_updates(model, updates)

        return model, new_state, loss, acts

    return compute_loss, step_model


@app.cell
def __(batch_size, beartype, eqx, jax, jaxtyped, jnp, n_steps, step_model):
    @jaxtyped(typechecker=beartype.beartype)
    def mean_acts_over_n_steps(
        model: eqx.Module | list[eqx.Module], optim, dataloader, *, seed: int
    ):
        state = optim.init(eqx.filter(model, eqx.is_inexact_array))

        mean_acts = []

        key = jax.random.key(seed)
        for b, batch in enumerate(dataloader):
            images = jnp.asarray(batch["image"])
            labels = jnp.asarray(batch["label"])

            key, *subkeys = jax.random.split(key, num=batch_size + 1)
            model, state, _, acts = step_model(
                model, optim, state, images, labels, keys=subkeys
            )
            mean_acts.append(jax.tree.map(lambda x: jnp.abs(x).mean(), acts))

            del images
            del labels

            if b + 1 >= n_steps:
                break

        return jax.tree.map(lambda *xs: jnp.array(xs), *mean_acts)

    return (mean_acts_over_n_steps,)


@app.cell
def __(
    MuParamVisionTransformer,
    eqx,
    jax,
    learning_rate,
    mean_acts_over_n_steps,
    mup_init,
    optax,
    standard_init,
    train_dataloader,
):
    def sweep_width(*, do_mup_init: bool, do_mup_lr: bool, seed: int):
        min_width = 6
        max_width = 12
        base_width = 2**min_width

        for i, width_exp in enumerate(range(min_width, max_width + 1)):
            width = 2**width_exp

            model_cfg = dict(
                d=width,
                hidden_d=width * 4,
                n_heads=4,
                n_layers=2,
                p_dropout=0.0,
                patch_size=16,
                n_patches=196,
                n_classes=1000,
            )

            m_d = width / base_width

            model = MuParamVisionTransformer(**model_cfg, key=jax.random.key(0))

            init_key = key = jax.random.key(seed=i + seed)
            if do_mup_init:
                model = mup_init(model, 0.02, m_d=m_d, key=init_key)
                # if width_exp == 8:
                #     breakpoint()
            else:
                model = standard_init(model, 0.02, key=init_key)

            param_spec = jax.tree.map(
                lambda _: "hidden", eqx.filter(model, eqx.is_inexact_array)
            )
            param_spec = eqx.tree_at(
                lambda m: m.patch_embedding.linear.weight, param_spec, "embedding"
            )
            param_spec = eqx.tree_at(
                lambda m: m.patch_embedding.linear.bias, param_spec, "embedding"
            )

            if do_mup_lr:
                lr = learning_rate / m_d
                optim = optax.multi_transform(
                    {
                        "hidden": optax.adam(
                            learning_rate=learning_rate / m_d,
                            b1=0.9,
                            b2=0.999,
                        ),
                        "embedding": optax.adam(
                            learning_rate=learning_rate,
                            b1=0.9,
                            b2=0.999,
                        ),
                    },
                    [param_spec],
                )
                model = [model]
            else:
                optim = optax.adam(
                    learning_rate=learning_rate,
                    b1=0.9,
                    b2=0.999,
                )

            # optim = optax.chain(optim, optax.clip_by_global_norm(1.0))

            mean_acts = mean_acts_over_n_steps(
                model, optim, train_dataloader, seed=i + seed
            )
            del model
            print(f"{width_exp} ({i + 1}/{max_width - min_width + 1}) done.")
            yield 2**width_exp, mean_acts

    return (sweep_width,)


@app.cell
def __(n_seeds, sweep_width):
    mup_init_mup_lr_data = [
        (width, mean_acts)
        for i in range(n_seeds)
        for width, mean_acts in sweep_width(do_mup_init=True, do_mup_lr=True, seed=i)
    ]
    return (mup_init_mup_lr_data,)


@app.cell
def __():
    # sp_data = [
    #     (width, mean_acts)
    #     for i in range(n_seeds)
    #     for width, mean_acts in sweep_width(do_mup_init=False, do_mup_lr=False, seed=i)
    # ]
    return


@app.cell
def __(beartype, chex, eqx, jax, jaxtyped):
    @jaxtyped(typechecker=beartype.beartype)
    def standard_init(model: eqx.Module, std: float, *, key: chex.PRNGKey):
        """
        Re-init all eqx.nn.Linear modules to have weights sampled from ~N(0, std^2)
        """

        is_linear = lambda x: isinstance(x, eqx.nn.Linear)
        get_weights = lambda m: [
            x.weight
            for x in jax.tree_util.tree_leaves(m, is_leaf=is_linear)
            if is_linear(x)
        ]
        weights = get_weights(model)
        new_weights = [
            jax.random.normal(subkey, weight.shape, weight.dtype) * std
            for weight, subkey in zip(weights, jax.random.split(key, len(weights)))
        ]
        new_model = eqx.tree_at(get_weights, model, new_weights)
        return new_model

    return (standard_init,)


@app.cell
def __(beartype, chex, eqx, jax, jaxtyped, jnp):
    @jaxtyped(typechecker=beartype.beartype)
    def mup_init(model: eqx.Module, std: float, *, m_d: float, key: chex.PRNGKey):
        """
        Re-init all eqx.nn.Linear modules to have weights sampled from ~N(0, std^2 / m_d).
        The Eluether [post](https://blog.eleuther.ai/mutransfer/) specifies not changing the embedding weight initialization.
        But that's for language.
        For vision, where our inputs are already dense, we want the patch embedding to also be reduced by m_d.
        """
        is_linear = lambda x: isinstance(x, eqx.nn.Linear)
        get_weights = lambda m: [
            x.weight
            for x in jax.tree_util.tree_leaves(m, is_leaf=is_linear)
            if is_linear(x)
        ]
        weights = get_weights(model)
        new_weights = [
            jax.random.normal(subkey, weight.shape, weight.dtype) * std / jnp.sqrt(m_d)
            for weight, subkey in zip(weights, jax.random.split(key, len(weights)))
        ]
        new_model = eqx.tree_at(get_weights, model, new_weights)
        return new_model

    return (mup_init,)


@app.cell
def __(get_df, make_chart, sp_data):
    sp_df = get_df(sp_data)

    make_chart(sp_df)
    return (sp_df,)


@app.cell
def __(n_seeds, sweep_width):
    mup_init_data = [
        (width, mean_acts)
        for i in range(n_seeds)
        for width, mean_acts in sweep_width(do_mup_init=True, do_mup_lr=False, seed=i)
    ]
    return (mup_init_data,)


@app.cell
def __(get_df, make_chart, mup_init_data):
    mup_init_df = get_df(mup_init_data)

    make_chart(mup_init_df)
    return (mup_init_df,)


@app.cell
def __(get_df, make_chart, mup_init_mup_lr_data):
    mup_init_mup_lr_df = get_df(mup_init_mup_lr_data)

    make_chart(mup_init_mup_lr_df)
    return (mup_init_mup_lr_df,)


@app.cell
def __(alt, jnp, n_steps, pl):
    fields = ("patch", "attn", "ffn", "logits")

    def get_df(data):
        rows = []
        for width, mean_acts in data:
            for timestep in range(n_steps):
                for field in fields:
                    rows.append({
                        "width": jnp.log2(width).item(),
                        "timestep": timestep + 1,
                        "field": field,
                        "value": getattr(mean_acts, field)[timestep],
                    })

        return pl.DataFrame(rows)

    def make_chart(df):
        base = (
            alt.Chart(df)
            .mark_line()
            .encode(
                x=alt.X("width:Q").title("log2(width)"),
                y=alt.Y("mean(value):Q").scale(type="log"),
                color="timestep:Q",
            )
            .properties(width=180, height=180)
        )

        base += (
            alt.Chart(df)
            .mark_point()
            .encode(
                x=alt.X("width:Q").title("log2(width)"),
                y=alt.Y("mean(value):Q").scale(type="log"),
                color="timestep:Q",
            )
            .properties(width=180, height=180)
        )

        row1 = alt.hconcat()
        for field in ("patch", "attn"):
            row1 |= base.properties(title=field.capitalize()).transform_filter(
                alt.datum.field == field
            )

        row2 = alt.hconcat()
        for field in ("ffn", "logits"):
            row2 |= base.properties(title=field.capitalize()).transform_filter(
                alt.datum.field == field
            )

        return row1 & row2

    return fields, get_df, make_chart


@app.cell
def __(mup_init_mup_lr_df):
    mup_init_mup_lr_df
    return


@app.cell
def __(mup_init_df):
    mup_init_df
    return


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()

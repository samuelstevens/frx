import marimo

__generated_with = "0.8.15"
app = marimo.App(width="full")


@app.cell
def __():
    import altair as alt
    import jax
    import jax.numpy as jnp
    import marimo as mo
    import matplotlib.pyplot as plt

    return alt, jax, jnp, mo, plt


@app.cell
def __(jax):
    d_in = 768
    sigma_x = 1.2

    x = jax.random.normal(jax.random.key(3), (d_in,)) * sigma_x
    return d_in, sigma_x, x


app._unparsable_cell(
    r"""
    d_out = 256
    sigma_W =  / jnp.sqrt(d_in)

    W = jax.random.normal(jax.random.key(1), (d_out, d_in)) * sigma_W
    """,
    name="__",
)


@app.cell
def __(W, x):
    y = W @ x
    return (y,)


@app.cell
def __(jnp, y):
    jnp.std(y)
    return


@app.cell
def __(d_in, jnp, sigma_W, sigma_x):
    jnp.sqrt(d_in * sigma_x**2 * sigma_W**2)
    return


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()

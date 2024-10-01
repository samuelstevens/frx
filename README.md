# FRX

FRX (pronounced "freaks") is a Python package for training vision models from scratch in Jax.

See the [API reference docs](https://samuelstevens.me/frx/).

## Motivation

I worked on [BioCLIP](https://imageomics.github.io/bioclip/), a foundation vision model for the entire tree of life (450K+ different species).
One of our ablations was comparing our CLIP-based objective to a traditional cross entropy classification objective.
I felt that we did not aggressively tune the cross entropy training hyperparameters, and that it is not a 100% perfectly fair comparison (but science is never 100% perfect--BioCLIP is still a phenomenal piece of work of which I am immensely proud).
I am curious if [muP](https://github.com/microsoft/mup?tab=readme-ov-file#checking-correctness-of-parametrization) (see [my notes on muP](https://samuelstevens.me/writing/mup)) along with Jax can lead to a stronger cross-entropy baseline.

With that in mind, one concrete research goal of FRX is to train a ViT-B/16 from scratch on [TreeOfLife-10M](https://huggingface.co/datasets/imageomics/TreeOfLife-10M) using a cross-entropy classification objective and compare it to BioCLIP on [biobench](https://github.com/samuelstevens/biobench).

Personally, I am also interested in learning more about:

* Jax, as well as multi-GPU and multi-node training in Jax.
* muP transfer, including [unit-scaled muP](https://arxiv.org/abs/2407.17465).
* Large(r)-scale vision training.

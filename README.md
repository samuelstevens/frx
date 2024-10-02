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

## Further Reading

* [This page](https://www.tensorflow.org/extras/candidate_sampling.pdf) describes candidate sampling techniques for classification pre-training with many classes.
* Generally, "extreme classification" describes classification problems with millions or billions of classes.

## Road Map

1. Demonstrate muP transfer of learning rate and std dev of weight initialization on ImageNet-1K using a model with size 128, 256 and 384 (384 is a ViT-S/16).
2. Do the same with TreeOfLife-10M.
3. Experiment with extreme classification methods.

## muP Transfer

To demonstrate that muP transfer actually works, we have to tune the larger model using standard parameterization as well.
However, we don't want to do this for actually big models--the whole point of muP is to avoid that.
So we demonstrate that muP works by tuning models with a hidden dimension of 128, 256 and 384 with 12 layers.
We also do muP tuning on the d=128 model then do zero-shot muP transfer to d=384 and compare it to the manually tuned d=384.

Experimental results are described [here](https://samuelstevens.me/frx/frx/mup) in the docs.

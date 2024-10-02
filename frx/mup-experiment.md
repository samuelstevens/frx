# Motivation

We want to empirically demonstrate that our muP implementation is correct.
To do so, we will do both standard hyperparameter tuning (SP) and muP hyperparameter tuning (muP) on three differently sized models up to a ViT-S/16 on ImageNet-1K.
These are not meant to achieve SOTA results; we merely want to show that muP enables zero-shot hyperparameter transfer to larger models so we don't have to tune larger models (like ViT-L/14).

# Experimental Setup

For each model size of d=128, d=256, d=384, we randomly sample 10 combinations of (1) learning rate and (2) weight initialization stdev.
The range for each hyperparameter is in `experiments/mup-demo/sp-baseline.toml`.
We train for 90 epochs on ImageNet-1K.

Then we train d=256 and d=384 models using muP transfer rules from d=128 and compare to the manually tuned versions. 
Those configurations are in `experiments/mup-demo/mup-transfer.toml`.

Experiments are tagged either `sp-baseline` or `mup-transfer` on [WandB](https://wandb.ai/samuelstevens/frx).
[This report](https://wandb.ai/samuelstevens/frx/reports/muP-Transfer--Vmlldzo5NTgzMDYz) contains a nice written summary.

# Results



# Discussion



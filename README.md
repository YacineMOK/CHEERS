# CHEERS
###### Yacine Mokhtari & Grégory Smits — Preprint

This repository contains the code for an ongoing research project.

CHEERS aims to learn shared criteria from item feature values using trainable prototypes. It aggregates their activations through personalized additive (CHEERS-1) or 2-additive Choquet (CHEERS-2) models, making criterion contributions and pairwise interactions available for inspection.

The following figure illustrates the proposed model pipeline.

![CHEERS model pipeline](./CHEERS_archi.jpg)


The repository is structured as follows:

- `./model/` contains the main model implementations.
- `./saved_model_params/` contains the saved CHEERS-1 and CHEERS-2 models trained on SynthAddOnly.
- `./utils/` contains data-loading, evaluation, and visualization utilities.
- `./datasets/` contains the generated SynthAddOnly dataset.
- `./dataset_generation.ipynb` provides the notebook for generating SynthAddOnly.
- `./model_training_inference.ipynb` provides the notebook for training, saving, loading, and running the models, and inspecting their learned representations and preference parameters.
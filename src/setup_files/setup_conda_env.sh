#!/bin/bash

# Create a conda env called "ml":
mamba create --prefix ~/scratch/cv -y python=3.10
mamba activate ~/scratch/cv

# Install PyTorch
mamba install -y pytorch torchvision torchaudio pytorch-cuda=12.1 \
    -c pytorch -c nvidia

# Install some other usual libraries
mamba install -y -c conda-forge numpy matplotlib scipy pandas \
    scikit-learn pillow optuna ipykernel jupyter notebook 

# Install utils here
pip install -e .

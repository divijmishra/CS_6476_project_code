#!/bin/bash

# mkdir -p ~/scratch/miniconda3
# wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/scratch/miniconda3/miniconda.sh
# bash ~/scratch/miniconda3/miniconda.sh -b -u -p ~/scratch/miniconda3
# rm -rf ~/scratch/miniconda3/miniconda.sh
# # ~/scratch/miniconda3/bin/conda init bash

mkdir -p ~/scratch/miniforge3
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" \
    -O ~/scratch/miniforge3/miniforge.sh
bash ~/scratch/miniforge3/miniforge.sh -b -u -p ~/scratch/miniforge3
rm -rf ~/scratch/miniforge3/miniforge.sh
~/scratch/miniforge3/bin/mamba init

#!/bin/bash

# create conda env
read -rp "Enter environment name (recommended: multilingual-srl): " env_name
read -rp "Enter python version (recommended: 3.8): " python_version
conda create -yn "$env_name" python="$python_version"
eval "$(conda shell.bash hook)"
conda activate "$env_name"

# install torch
read -rp "Enter cuda version (e.g. '11.3' or 'none' to avoid installing cuda support): " cuda_version
if [ "$cuda_version" == "none" ]; then
    conda install -y pytorch=1.12 cpuonly -c pytorch
else
    conda install -y pytorch=1.12 cudatoolkit=$cuda_version -c pytorch -c conda-forge
fi

# install dependencies
conda install -y pytorch-scatter -c pyg

# install requirements
pip install -r requirements.txt
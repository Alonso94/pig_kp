#!/bin/bash
# Create the conda environment from the file
conda env create --file pig.yml
eval "$(conda shell.bash hook)"
# Activate the environment
conda activate pig
# Install the package
pip install -e .
# install the extensions
python setup_ext.py install
#!/bin/sh
module load anaconda3/personal
module load cuda/10.2
conda env create -f environment.yml

conda deactivate

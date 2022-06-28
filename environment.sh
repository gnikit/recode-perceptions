#!/bin/sh
module load anaconda3/personal
module load cuda/10.2
conda create -n recode python=3.7
source activate recode

conda install pytorch==1.9.1 torchvision==0.10.1 torchaudio==0.9.1 cudatoolkit=10.2 -c pytorch
conda install -c conda-forge wandb
conda install -c conda-forge tqdm
conda install pandas
conda install matplotlib

conda deactivate

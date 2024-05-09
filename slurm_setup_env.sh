#!/bin/bash

#SBATCH --mem-per-cpu=3000
#SBATCH -A research
#SBATCH -n 39
#SBATCH --gres=gpu:4
#SBATCH -t 4-00:00:00
#SBATCH --mail-type=NONE
#SBATCH --job-name="TorRNA Environment"
#SBATCH --output=torrna.txt

ulimit -n 40960

source "/home2/$USER/miniconda3/etc/profile.d/conda.sh"
source "/home2/$USER/miniconda3/etc/profile.d/mamba.sh"

rm -r "/scratch/$USER/"
mkdir -p "/scratch/$USER/conda_pkgs_dirs"
export CONDA_PKGS_DIRS="/scratch/$USER/conda_pkgs_dirs"

rm -r "/scratch/$USER/torrna/"
rm -r "/scratch/$USER/pkgs_dirs"
mamba create --prefix "/scratch/$USER/torrna" python=3.8 --yes
mamba activate "/scratch/$USER/torrna"
module load u18/cuda/11.6

pip install rna-fm==0.2.2 tqdm ptflops biopython ml_collections wandb matplotlib ipykernel rdkit-pypi seaborn
# pip install rna-fm tqdm ptflops biopython ml_collections wandb matplotlib ipykernel rdkit-pypi seaborn torch==1.11.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html


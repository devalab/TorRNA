# TorRNA
Improved prediction of Torsion angles of RNA by leveraging large language models.

## Setting up dependencies 

The script `slurm_setup_env.sh` contains the commands to set up a minimal Anaconda environment on a HPC with a Slurm workload manager, the CUDA 11.6 toolkit can be loaded with the `module load...` command that is commented out in the script. The script can be run as follows:

```
source slurm_setup_env.sh
```

## Data

The `./data` folder has all the data required to train and test TorRNA. 

## Training TorRNA

`main.py` trains a model of TorRNA and saves model checkpoints in the `./checkpoints` folder. We have provided the best TorRNA models in this folder.

## Testing TorRNA

`compare_predictions.py` tests TorRNA on the test set, and also gives the prediction errors for SPOT-RNA-1D and a random baseline method.

### Jupyter Notebooks

All the results in the manuscript were generated using the `.ipynb` files provided in this repository.

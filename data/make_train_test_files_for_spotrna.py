import os
import torch
import numpy as np
import shutil
import pickle
import glob
from pathlib import Path

from tqdm import tqdm
from multiprocessing import Pool
from itertools import repeat

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from data.rna_object import RNA


all_spot_files_path = "/home2/sriram.devata/rna_project/rna_structure/data/raw_files/purified_spot_pdbs/necessary_spot_pdbs"

all_pdbs = list()
for each_pdb in glob.glob(f"{all_spot_files_path}/*.pdb"):
    each_pdb = each_pdb.strip()
    pdb_name = each_pdb.split("/")[-1].replace(".pdb", '')
    

    try:
        pdb_path = f"{all_spot_files_path}/{pdb_name}.pdb"
        print(pdb_path)
        loaded_rna = RNA(pdb_path, calc_rna_fm_embeddings=False, load_dssr_dihedrals=True, load_coords=False, dssr_path="/home2/sriram.devata/rna_project/rna_structure/data/raw_files/purified_spot_pdbs/necessary_spot_pdbs")
        if len(loaded_rna.full_seq) < 500 and len(loaded_rna.full_seq) > 1:
            all_pdbs.append(pdb_name)
        if len(loaded_rna.dssr_full_seq) < 500 and len(loaded_rna.dssr_full_seq) > 1:
            all_pdbs.append(pdb_name)

    except Exception as e:
        print(e, pdb_name)
        pass

spot_dataset_files_path = "/home2/sriram.devata/rna_project/cdhit/spot-rna-1d-datasets"

dataset_files = ["TR.tsv", "TS1.tsv", "TS2.tsv", "TS3.tsv", "VL.tsv"]
training_full_pdbs = list()
testing_full_pdbs = list()

for each_dataset_file in dataset_files:
    with open(f"{spot_dataset_files_path}/{each_dataset_file}", 'r') as f:
        all_lines = f.readlines()

    cur_dataset_pdbs = [x.strip().upper() for x in all_lines]# list(set([x.strip().split('_')[0] for x in all_lines]))

    if "TR" in each_dataset_file:
        training_full_pdbs += cur_dataset_pdbs
    else:
        testing_full_pdbs += cur_dataset_pdbs

print(len(all_pdbs))
        
final_training_set = list()
final_test_set = list()
for each_pdb in all_pdbs:
    if each_pdb.upper() in training_full_pdbs:
        final_training_set.append(each_pdb)
    else:
        final_test_set.append(each_pdb)

final_training_set = list(set(final_training_set))
final_test_set = list(set(final_test_set))
            
print(f"#{len(final_training_set)} training PDBs, #{len(final_test_set)} testing PDBs")

with open(f"data/spot_pdb_files_train_test.pkl", "wb") as fp:
    pickle.dump((final_training_set, final_test_set), fp)

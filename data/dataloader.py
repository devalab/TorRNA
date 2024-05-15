import os
import torch
import numpy as np
import shutil
import pickle
from pathlib import Path

from tqdm import tqdm
from multiprocessing import Pool
from itertools import repeat

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from data.rna_object import RNA

def old_make_good_pdbs_list(pdbs_path, processed_dir):

    if os.path.exists(f"{processed_dir}/good_pdb_files.pkl"):
        print(f"Found precomputed good PDB files")
        with open(f"{processed_dir}/good_pdb_files.pkl", "rb") as fp:
            good_pdb_files = pickle.load(fp)
        return good_pdb_files

    all_pdbs_file = f"{pdbs_path}/all_pdbs.txt"
    with open(all_pdbs_file, "r") as f:
        all_pdbs_lines = f.readlines()

    all_pdbs = list()
    for each_line in all_pdbs_lines:
        each_line = each_line.strip().split(",")
        for each_pdb in each_line:
            all_pdbs.append(f"{each_pdb}.pdb")

    good_pdb_files = list()
    print(f"Calculating good PDB files:")
    pb = tqdm(all_pdbs)
    for each_pdb in pb:
        try:
            pdb_path = f"{pdbs_path}/{each_pdb}"
            loaded_rna = RNA(pdb_path, calc_rna_fm_embeddings=False, load_dssr_dihedrals=True, load_coords=False)
            if len(loaded_rna.full_seq) < 500 and len(loaded_rna.full_seq) > 1:
                good_pdb_files.append(pdb_path)
            if len(loaded_rna.dssr_full_seq) < 500 and len(loaded_rna.dssr_full_seq) > 1:
                good_pdb_files.append(pdb_path)

            pb.set_description(f"Good PDBs: {len(good_pdb_files)}")
        except Exception as e:
            # print(e, each_pdb)
            pass

    import random
    random.shuffle(good_pdb_files)

    with open(f"{processed_dir}/good_pdb_files.pkl", "wb") as fp:
        pickle.dump((good_pdb_files), fp)

    return good_pdb_files


def add_to_RNATorsionalAnglesDataset(each_pdb_file):
    rna_object = RNA(each_pdb_file)
    rna_fm_embeddings = rna_object.rna_fm_embeddings
    torsional_angles = rna_object.torsion_angles
    initial_embeddings = rna_object.rna_fm_initial_embeddings

    dataset_list.append((rna_fm_embeddings, torsional_angles, initial_embeddings))

class RNATorsionalAnglesDataset(Dataset):
    def __init__(self, all_pdb_files, processed_dir, type_dataset="train"):

        self.all_pdb_files = all_pdb_files

        if os.path.exists(f"{processed_dir}/{type_dataset}_dataset.pkl"):
            print(f"Found precomputed {type_dataset} dataset")
            with open(f"{processed_dir}/{type_dataset}_dataset.pkl", "rb") as fp:
                # self.all_rna_fm_embeddings, self.all_torsional_angles, self.all_dssr_torsional_angles, self.all_initial_embeddings = pickle.load(fp)
                self.all_rna_fm_embeddings, self.all_dssr_torsional_angles, self.all_initial_embeddings = pickle.load(fp)

        else:
            self.all_rna_fm_embeddings = list()
            self.all_torsional_angles = list()
            self.all_initial_embeddings = list()
            self.all_dssr_torsional_angles = list()

            # self.all_seq_one_hot = list()
            # self.len_of_seq = list()

            print(f"Making RNATorsionalAnglesDataset...")
            for each_pdb_file in tqdm(self.all_pdb_files):
                try:
                    rna_object = RNA(each_pdb_file, calc_rna_fm_embeddings=True, load_dssr_dihedrals=True, load_coords=False)
                    self.all_rna_fm_embeddings.append(rna_object.rna_fm_embeddings)
                    # self.all_torsional_angles.append(rna_object.torsion_angles)
                    self.all_initial_embeddings.append(rna_object.rna_fm_initial_embeddings)
                    self.all_dssr_torsional_angles.append(rna_object.dssr_torsion_angles)
                    # self.all_seq_one_hot.append(rna_object.all_one_hots)
                    # self.len_of_seq.append(torch.tensor(len(rna_object.all_one_hots)))
                except Exception as e:
                    print(f"Could not load {each_pdb_file} due to {e}")

            with open(f"{processed_dir}/{type_dataset}_dataset.pkl", "wb") as fp:
                # pickle.dump((self.all_rna_fm_embeddings, self.all_torsional_angles, self.all_dssr_torsional_angles, self.all_initial_embeddings), fp)
                pickle.dump((self.all_rna_fm_embeddings, self.all_dssr_torsional_angles, self.all_initial_embeddings), fp)

    def __len__(self):
        return len(self.all_dssr_torsional_angles)

    def __getitem__(self, idx):
        # return self.all_rna_fm_embeddings[idx], self.all_torsional_angles[idx], self.all_dssr_torsional_angles[idx], self.all_initial_embeddings[idx]
        return self.all_rna_fm_embeddings[idx], self.all_dssr_torsional_angles[idx], self.all_initial_embeddings[idx]


def collate_fn(data):
    rna_fm_embeddings, torsional_angles, initial_embeddings = zip(*data)

    rna_fm_embeddings = pad_sequence(rna_fm_embeddings, batch_first=True) # shape - (num_batch, max_len_rna+2, 640)
    torsional_angles = pad_sequence(torsional_angles, batch_first=True) # shape - (num_batch, max_len_rna, 18)
    initial_embeddings = pad_sequence(initial_embeddings, batch_first=True) # shape - (num_batch, max_len_rna+2, 640)
    # seq_one_hot = pad_sequence(seq_one_hot, batch_first=True)   # shape - (num_batch, max_len_rna, 4)
    # len_of_seqs = torch.stack(len_of_seqs)  # shape - (num_batch)

    padding_mask = torch.sum(torch.abs(torsional_angles), dim=-1) < 1e-5   # shape - (num_batch, max_len_rna), False when it is an actual value and True when it is padded
    # TODO check how this mask handles the nan values in torsional_angles

    return rna_fm_embeddings, torsional_angles, initial_embeddings, padding_mask


def get_train_test_dataloaders(root="/ssd_scratch/users/sriram.devata/rna_structure/dataset/",
                            pdbs_path="/home2/sriram.devata/rna_project/rna_structure/data/raw_files/all_torrna_pdbs/",
perfect_pdb_files_train_val_test_path="/home2/sriram.devata/rna_project/cdhit/torrna_train_val_test.pkl"):

    processed_dir = f"{root}/"
    Path(processed_dir).mkdir(parents=True, exist_ok=True)

    with open(perfect_pdb_files_train_val_test_path, "rb") as fp:
        training_pdbs, validation_pdbs, testing_pdbs = pickle.load(fp)
    training_files = list()
    for each_training_pdb in training_pdbs:
        training_files.append(f"{pdbs_path}/{each_training_pdb}.pdb")
    validation_files = list()
    for each_validation_pdb in validation_pdbs:
        validation_files.append(f"{pdbs_path}/{each_validation_pdb}.pdb")
    testing_files = list()
    for each_testing_pdb in testing_pdbs:
        testing_files.append(f"{pdbs_path}/{each_testing_pdb}.pdb")


    train_dataset = RNATorsionalAnglesDataset(training_files, processed_dir=processed_dir, type_dataset="train")
    val_dataset = RNATorsionalAnglesDataset(validation_files, processed_dir=processed_dir, type_dataset="val")
    test_dataset = RNATorsionalAnglesDataset(testing_files, processed_dir=processed_dir, type_dataset="test")

    train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=32)
    val_dataloader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=32)
    test_dataloader = DataLoader(test_dataset, collate_fn=collate_fn, batch_size=32)


    return train_dataloader, val_dataloader, test_dataloader


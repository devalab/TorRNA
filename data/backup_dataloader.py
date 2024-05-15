import os
import torch
import numpy as np
import shutil

from tqdm import tqdm
from multiprocessing import Pool
from itertools import repeat

from torch_geometric.data import InMemoryDataset, Dataset
from torch_geometric.loader import DataLoader

from data.rna_object import RNA

def make_data_splits(pdbs_path):

    all_pdbs_file = f"{pdbs_path}/all_pdbs.txt"
    with open(all_pdbs_file, "r") as f:
        all_pdbs_lines = f.readlines()

    all_pdbs = list()
    for each_line in all_pdbs_lines:
        each_line = each_line.strip().split(",")
        for each_pdb in each_line:
            all_pdbs.append(f"{each_pdb}.pdb")

    # calculate the total number of datapoints that can be obtained from all PDBs
    num_datapoints = list()
    good_pdb_files = list()
    print(f"Calculating data splits:")
    for each_pdb in tqdm(all_pdbs):
        try:
            pdb_path = f"{pdbs_path}/{each_pdb}"
            loaded_rna = RNA(pdb_path)
            if len(loaded_rna.coords_list_of_residues) < 500:
                num_datapoints.append(len(loaded_rna.coords_list_of_residues) - 1)
                good_pdb_files.append(pdb_path)
        except:
            pass

    # divide all PDBs into 20 sets
    # all_splits will have - [0, 43, 86, 129, 172, 215, 258..., 861] where each elem has the split's cumulative number of valid PDBs for example
    all_splits = list(range(0, len(num_datapoints), int(len(num_datapoints)/20)))
    if all_splits[-1] != len(num_datapoints)-1:
        all_splits.append(len(num_datapoints) - 1)

    # get the total number of datapoints in each of these 20 sets
    # all_splits_num_datapoints will have - [837, 3610, 808, 3203, 5650, ..., 14] for example
    all_splits_num_datapoints = list()
    for idx,each_split in enumerate(all_splits):
        if idx == 0:
            continue
        cur_split_num_datapoints = 0
        for each_rna_num_datapoints in num_datapoints[all_splits[idx-1]:each_split]:
            cur_split_num_datapoints += each_rna_num_datapoints
        all_splits_num_datapoints.append(cur_split_num_datapoints)

    # get the cumulative datapoints at the end of each of these 20 sets
    # cumulative_datapoint_ids will have - [837, 4447, 5255, 8458, ..., 56090] for example
    # this shows that the split 1 will have till datapoint_id 836 for example
    cumulative_datapoint_ids = [all_splits_num_datapoints[0]]
    for each_split_num_datapoints in all_splits_num_datapoints[1:]:
        cumulative_datapoint_ids.append(cumulative_datapoint_ids[-1] + each_split_num_datapoints)

    return good_pdb_files, all_splits, cumulative_datapoint_ids


class RNAAutoRegressiveDataset(Dataset):
    def __init__(self, root, pdbs_path="/home2/sriram.devata/rna_structure/data/raw_files/all_pdbs",
                transform=None, pre_transform=None, pre_filter=None):

        self.pdbs_path = pdbs_path

        self.all_raw_file_names, self.pdb_splits, self.ids_in_data_splits = make_data_splits(pdbs_path)

        print(self.ids_in_data_splits)

        super().__init__(root, transform, pre_transform, pre_filter)
        # self.data, self.slices = torch.load(self.processed_paths[0])


    @property
    def raw_file_names(self):
        return self.all_raw_file_names
    #     all_pdbs_file = f"{self.pdbs_path}/all_pdbs.txt"
    #     with open(all_pdbs_file, "r") as f:
    #         all_pdbs_lines = f.readlines()

    #     all_pdbs = list()
    #     for each_line in all_pdbs_lines:
    #         each_line = each_line.strip().split(",")
    #         for each_pdb in each_line:
    #             all_pdbs.append(f"{each_pdb}.pdb")

    #     return all_pdbs

    @property
    def processed_file_names(self):

        # all_processed = list()
        # for each_pdb in self.raw_file_names:
        #     each_pdb = each_pdb.replace(".pdb", "")
        #     processed_file = f"{each_pdb}.pt"
        #     all_processed.append(processed_file)

        # return all_processed

        return [f"data_split_{idx}.pt" for idx in range(len(self.ids_in_data_splits))]

    def download(self):
        # Download to `self.raw_dir`.
        print(f"Copying {len(self.raw_file_names)} raw files")
        for each_raw_file in tqdm(self.raw_file_names):
            src_loc = f"{self.pdbs_path}/{each_raw_file}"
            dst_loc = f"{self.raw_dir}/{each_raw_file}"
            shutil.copyfile(src_loc, dst_loc)

    def process(self):

        #with Pool(5) as p:
        #    # p.map(process_dataset_split, range(1, len(self.pdb_splits)))
        #    p.starmap(process_dataset_split,
        #                zip(range(1, len(self.pdb_splits)),
        #                    repeat(self.pdb_splits),
        #                    repeat(self.raw_file_names),
        #                    repeat(self.pre_filter),
        #                    repeat(self.pre_transform),
        #                    repeat(self.processed_dir)))
        for split_idx in range(1, len(self.pdb_splits)):
            process_dataset_split(split_idx, self.pdb_splits,
                                    self.raw_file_names, self.pre_filter,
                                    self.pre_transform, self.processed_dir)

        # data_list = list()
        # print(f"Processing {len(self.raw_file_names)} PDB files")
        # for each_pdb in tqdm(self.raw_file_names):
        #     # print(each_pdb)
            
        #     try:
        #         pdb_path = f"{self.raw_dir}/{each_pdb}"
        #         loaded_rna = RNA(pdb_path)

        #         for resnum_idx in range(1, len(loaded_rna.coords_list_of_residues)):
        #             data_list.append(loaded_rna.make_graph_till_resnum(resnum_idx))
        #     except Exception as e:
        #         print(f"Error when processing {pdb_path}")

        # if self.pre_filter is not None:
        #     data_list = [data for data in data_list if self.pre_filter(data)]

        # if self.pre_transform is not None:
        #     data_list = [self.pre_transform(data) for data in data_list]

        # data, slices = self.collate(data_list)
        # torch.save((data, slices), self.processed_paths[0])

    def len(self):
        return self.ids_in_data_splits[-1]

    def get(self, idx):

        for split_id, cum_in_split in enumerate(self.ids_in_data_splits):
            if idx < cum_in_split:
                
                # load the entire split
                split_data = torch.load(os.path.join(self.processed_dir, f"data_split_{split_id}.pt"))

                # calculate the actual index of this idx inside the split_data
                if split_id == 0:
                    return split_data[idx]
                else:
                    return split_data[idx - self.ids_in_data_splits[split_id - 1]]
                return data


def process_dataset_split(split_idx, pdb_splits, raw_file_names, pre_filter, pre_transform, processed_dir):
    data_list = list()
    # load the PDBs that are supposed to be in this data split
    for pdb_idx in tqdm(range(pdb_splits[split_idx-1], pdb_splits[split_idx])):

        if pdb_idx % 1000 == 0:
            print(f"Progress when processing dataset split {split_idx}: {pdb_idx}/{pdb_splits[split_idx]}", flush=True)

        pdb_path = raw_file_names[pdb_idx]
        loaded_rna = RNA(pdb_path)

        for resnum_idx in range(1, len(loaded_rna.coords_list_of_residues)):
            data_list.append(loaded_rna.make_graph_till_resnum(resnum_idx))

    if pre_filter is not None:
        data_list = [data for data in data_list if pre_filter(data)]

    if pre_transform is not None:
        data_list = [pre_transform(data) for data in data_list]
    
    torch.save(data_list, os.path.join(processed_dir, f"data_split_{split_idx-1}.pt"))
    print(f"Finished processing dataset split {split_idx}", flush=True)

def get_train_test_dataloaders(root="/ssd_scratch/users/sriram.devata/rna_structure/dataset",
                            pdbs_path="/home2/sriram.devata/rna_structure/data/raw_files/purified_small_pdbs/"):
    full_dataset = RNAAutoRegressiveDataset(root=root, pdbs_path=pdbs_path)
    # full_dataset = RNAAutoRegressiveDataset(root="/ssd_scratch/users/sriram.devata/rna_structure/dataset", pdbs_path="/home2/sriram.devata/rna_structure/data/raw_files/purified_small_pdbs/")
    full_dataset = full_dataset.shuffle()
    num_train_samples = int(0.8*len(full_dataset))
    train_loader = DataLoader(full_dataset[:num_train_samples], batch_size=256, shuffle=True, num_workers=20)
    test_loader = DataLoader(full_dataset[num_train_samples:], batch_size=128, shuffle=True, num_workers=10)

    return train_loader, test_loader


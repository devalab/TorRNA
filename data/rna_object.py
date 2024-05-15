import numpy as np
import os
import math

import torch
from torch.nn import PairwiseDistance

import fm
from copy import deepcopy

from Bio.PDB import PDBParser
from data.constants import RNA_CONSTANTS

from data.rigid_utils import Rotation, Rigid, calc_rot_tsl, calc_angl_rot_tsl, merge_rot_tsl, calc_dihedral

from utils import one_of_k_encoding

# Load RNA-FM model
rna_fm_model, rna_fm_alphabet = fm.pretrained.rna_fm_t12()
rna_fm_model.eval()  # disables dropout for deterministic results
rna_fm_batch_converter = rna_fm_alphabet.get_batch_converter()

class RNAFMEmbedder():
    def __init__(self, rna_fm_model_instance):
        self.embed_tokens = deepcopy(rna_fm_model.embed_tokens)
        self.embed_scale = deepcopy(rna_fm_model.embed_scale)
        self.embed_positions = deepcopy(rna_fm_model.embed_positions)

    def get_embedding(self, tokens):
        initial_embeddings = self.embed_scale * self.embed_tokens(tokens)
        initial_embeddings = initial_embeddings + self.embed_positions(tokens)

        return initial_embeddings
        
rna_fm_embedder = RNAFMEmbedder(rna_fm_model)

torsional_and_pseudo_torsional_angles = {
    "A" : {
        "alpha" : ["OP2", "P", "O5'", "C5'"],
        "beta" : ["P", "O5'", "C5'", "C4'"],
        "gamma" : ["O5'", "C5'", "C4'", "C3'"],
        "delta" : ["C5'", "C4'", "C3'", "O3'"],
        "chi" : ["C2'", "C1'", "N9", "C4"],
        "epsilon" : ["C4'", "C3'", "O3'", "Pi1"],
        "zeta" : ["C3'", "O3'", "Pi1", "O5'i1"],
        "eta" : ["OP2", "P", "C4'", "Pi1"],
        "theta" : ["P", "C4'", "Pi1", "OP2i1"]
    },
    "G" : {
        "alpha" : ["OP2", "P", "O5'", "C5'"],
        "beta" : ["P", "O5'", "C5'", "C4'"],
        "gamma" : ["O5'", "C5'", "C4'", "C3'"],
        "delta" : ["C5'", "C4'", "C3'", "O3'"],
        "chi" : ["C2'", "C1'", "N9", "C4"],
        "epsilon" : ["C4'", "C3'", "O3'", "Pi1"],
        "zeta" : ["C3'", "O3'", "Pi1", "O5'i1"],
        "eta" : ["OP2", "P", "C4'", "Pi1"],
        "theta" : ["P", "C4'", "Pi1", "OP2i1"]
    },
    "U" : {
        "alpha" : ["OP2", "P", "O5'", "C5'"],
        "beta" : ["P", "O5'", "C5'", "C4'"],
        "gamma" : ["O5'", "C5'", "C4'", "C3'"],
        "delta" : ["C5'", "C4'", "C3'", "O3'"],
        "chi" : ["C2'", "C1'", "N1", "C2"],
        "epsilon" : ["C4'", "C3'", "O3'", "Pi1"],
        "zeta" : ["C3'", "O3'", "Pi1", "O5'i1"],
        "eta" : ["OP2", "P", "C4'", "Pi1"],
        "theta" : ["P", "C4'", "Pi1", "OP2i1"]
    },
    "C" : {
        "alpha" : ["OP2", "P", "O5'", "C5'"],
        "beta" : ["P", "O5'", "C5'", "C4'"],
        "gamma" : ["O5'", "C5'", "C4'", "C3'"],
        "delta" : ["C5'", "C4'", "C3'", "O3'"],
        "chi" : ["C2'", "C1'", "N1", "C2"],
        "epsilon" : ["C4'", "C3'", "O3'", "Pi1"],
        "zeta" : ["C3'", "O3'", "Pi1", "O5'i1"],
        "eta" : ["OP2", "P", "C4'", "Pi1"],
        "theta" : ["P", "C4'", "Pi1", "OP2i1"]
    },
}

class RNA:
    def __init__(self, pdb_path, calc_rna_fm_embeddings=True, load_dssr_dihedrals=True, dssr_path=None, load_coords=True, from_seq=None):

        self.pdb_path = pdb_path
        self.full_seq = ""
        self.dssr_full_seq = ""

        if load_coords:
            # Load the PDB and store the loaded structure of the RNA
            parser = PDBParser(QUIET=True)
            self.structure = parser.get_structure('struct', pdb_path)
            self.load_coords()
            self.make_torsion_angles()
            # self.make_one_hot()

        if load_dssr_dihedrals:
            self.make_dssr_torsion_angles_and_seq(pdb_path, dssr_path)

        if from_seq is not None:
            self.dssr_full_seq = from_seq

        if calc_rna_fm_embeddings:
            if load_coords:
                seq_data = [("", self.full_seq),]
            else:
                seq_data = [("", self.dssr_full_seq),]
            batch_labels, batch_strs, batch_tokens = rna_fm_batch_converter(seq_data)
            # Extract embeddings (on CPU)
            with torch.no_grad():
                results = rna_fm_model(batch_tokens, repr_layers=[12])
            token_embeddings = results["representations"][12]

            self.rna_fm_embeddings = token_embeddings[0] # shape (num_residues+2, 640)
            self.rna_fm_initial_embeddings = rna_fm_embedder.get_embedding(batch_tokens)[0] # shape (num_residues+2, 640)

        #     print(f"RNA Object: rna_fm_embeddings: {self.rna_fm_embeddings.shape}")
        # print(f"RNA Object: full_seq: {len(self.full_seq)}")
        # print(f"RNA Object: torsion_angles: {self.torsion_angles.shape}")
        # print(f"RNA Object: all_one_hots: {self.all_one_hots.shape}")

    def load_coords(self):

        for model in self.structure:
            # coords_list_of_residues is an ordered list of all residues and the coordinates of atoms in those residues
            self.coords_list_of_residues = list()
            self.full_seq = list()
            for chain in model:
                for residue in chain:

                    # residue_coords is a dictionary that will contain the coords of all atoms in this residue
                    residue_coords = dict()
                    residue_coords["resname"] = residue.get_resname()

                    if residue_coords["resname"] not in RNA_CONSTANTS.RESD_NAMES:
                        continue

                    self.full_seq += residue_coords["resname"]

                    for atom in residue:
                        # residue_coords[str(atom_name)] = tensor of shape (3)
                        residue_coords[atom.get_fullname().strip()] = torch.tensor(atom.get_coord())
                    
                    # TODO the first residue does not have P, OP1, OP2, REMOVE DUMMY COORDINATES, SHOULD EXTRAPOLATE
                    if "P" not in residue_coords:
                        residue_coords["P"] = torch.zeros(3)
                        residue_coords["OP1"] = torch.zeros(3)
                        residue_coords["OP2"] = torch.zeros(3)

                    self.coords_list_of_residues.append(residue_coords)
            
        self.coords = torch.zeros((len(self.coords_list_of_residues), len(RNA_CONSTANTS.ATOM_NAMES_PER_RESD["G"]), 3))  # make a tensor of coords of all residues, G residue has the most number of atoms

        # saving all the coords in a single tensor
        for res_idx, residue_coords_dict in enumerate(self.coords_list_of_residues):
            atom_order = RNA_CONSTANTS.ATOM_NAMES_PER_RESD[residue_coords_dict["resname"]]
            res_coords = list()
            for each_atom in atom_order:
                res_coords.append(residue_coords_dict[each_atom])
            res_coords = torch.stack(res_coords, dim=0)

            self.coords[res_idx,:res_coords.shape[0],:] = res_coords

    def make_torsion_angles(self):

        all_res_torsions = list()

        for res_idx, each_residue in enumerate(self.coords_list_of_residues):
            if res_idx == len(self.coords_list_of_residues) - 1:
                break
                
            res_name = each_residue["resname"]

            # each_res_torsions = [torch.tensor([torch.cos(torch.tensor([0])), torch.sin(torch.tensor([0]))]),
            #                         torch.tensor([torch.cos(torch.tensor([0])), torch.sin(torch.tensor([0]))])]
            each_res_torsions = list()

            def get_cur_or_next_res_atom_coords(atom_name):
                if "i1" in atom_name:
                    return self.coords_list_of_residues[res_idx+1][atom_name.replace("i1", "")]
                else:
                    return each_residue[atom_name]

            angle_defs = torsional_and_pseudo_torsional_angles[res_name]
            for angle_name in angle_defs:
                atom_names = angle_defs[angle_name]
                torsion_angle = calc_dihedral(get_cur_or_next_res_atom_coords(atom_names[0]),
                                            get_cur_or_next_res_atom_coords(atom_names[1]),
                                            get_cur_or_next_res_atom_coords(atom_names[2]),
                                            get_cur_or_next_res_atom_coords(atom_names[3])
                )
                each_res_torsions.append(torch.tensor([torch.cos(torsion_angle), torch.sin(torsion_angle)]))


            each_res_torsions = torch.stack(each_res_torsions, dim=0)
            all_res_torsions.append(each_res_torsions)

        all_res_torsions = torch.stack(all_res_torsions, dim=0).reshape(-1, 18)
        self.torsion_angles = all_res_torsions

    def make_dssr_torsion_angles_and_seq(self, pdb_path, dssr_path):
        dssr_dihedrals_path = "/home2/sriram.devata/rna_project/rna_structure/data/raw_files/all_torrna_pdbs/"

        pdb_name = pdb_path.split("/")[-1].replace(".pdb", "")
        if dssr_path is None:
            dssr_log_file = f"{dssr_dihedrals_path}/{pdb_name}.tor"
        else:
            dssr_log_file = f"{dssr_path}/{pdb_name}.tor"

        with open(dssr_log_file, "r") as f:
            dssr_log_file_lines = f.readlines()
                        
        # print(dssr_log_file)

        star_lines = list()
        for line_num, line in enumerate(dssr_log_file_lines):
            if "***" in line:
                star_lines.append(line_num)
            elif "base" in line and "alpha" in line and "beta" in line and "gamma" in line and "delta" in line:
                first_set_of_angles_start = line_num
            elif "base" in line and "eta" in line and "theta" in line:
                second_set_of_angles_start = line_num
        _, first_set_of_angles_end, second_set_of_angles_end = star_lines
        
        def make_degree_angle_float_rad(angle_str):
            if angle_str == "---":
                return float('nan')
            return float(angle_str) * math.pi / 180
                
        
        all_angles = list()
        for each_angle in torsional_and_pseudo_torsional_angles["A"]:
            all_angles.append(list())
        self.dssr_full_seq = ""
        for line in dssr_log_file_lines[first_set_of_angles_start+1:first_set_of_angles_end]:

            if len(line.strip().split()) == 12: # the A/S field after chi_angle can be empty
                _, base_name, chi_angle, _, alpha_angle, beta_angle, gamma_angle, delta_angle, epsilon_angle, zeta_angle, _, _ = line.strip().split()
            else:
                _, base_name, chi_angle, alpha_angle, beta_angle, gamma_angle, delta_angle, epsilon_angle, zeta_angle, _, _ = line.strip().split()

            all_angles[0].append(make_degree_angle_float_rad(alpha_angle))
            all_angles[1].append(make_degree_angle_float_rad(beta_angle))
            all_angles[2].append(make_degree_angle_float_rad(gamma_angle))
            all_angles[3].append(make_degree_angle_float_rad(delta_angle))
            all_angles[4].append(make_degree_angle_float_rad(chi_angle))
            all_angles[5].append(make_degree_angle_float_rad(epsilon_angle))
            all_angles[6].append(make_degree_angle_float_rad(zeta_angle))

            self.dssr_full_seq += base_name[-1].upper()
            
        for line in dssr_log_file_lines[second_set_of_angles_start+1:second_set_of_angles_end]:
            line = line.strip().split()
            _, _, eta_angle, theta_angle, _, _, _, _ = line
            all_angles[7].append(make_degree_angle_float_rad(eta_angle))
            all_angles[8].append(make_degree_angle_float_rad(theta_angle))
            
        all_res_torsions = list()
        for each_res_idx in range(len(all_angles[0])):
            each_res_torsions = list()
            for torsion_angle_idx in range(len(all_angles)):
                torsion_angle = torch.tensor(all_angles[torsion_angle_idx][each_res_idx])
                each_res_torsions.append(torch.tensor([torch.cos(torsion_angle), torch.sin(torsion_angle)]))
            each_res_torsions = torch.stack(each_res_torsions, dim=0)
            all_res_torsions.append(each_res_torsions)
            
        all_res_torsions = torch.stack(all_res_torsions, dim=0).reshape(-1, 18)
        self.dssr_torsion_angles = all_res_torsions

    def make_one_hot(self):

        self.all_one_hots = list()
        for res_idx, each_residue in enumerate(self.full_seq):
            resname_one_hot = torch.tensor(one_of_k_encoding(each_residue, RNA_CONSTANTS.RESD_NAMES), dtype=torch.float) # tensor of shape (4)
            
            self.all_one_hots.append(resname_one_hot)

        self.all_one_hots = torch.stack(self.all_one_hots, dim=0)


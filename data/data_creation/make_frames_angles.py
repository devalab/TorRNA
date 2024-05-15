import numpy as np
import torch
from Bio.PDB import PDBParser, internal_coords
from constants import RNA_CONSTANTS

from rigid_utils import Rotation, Rigid, calc_rot_tsl, calc_angl_rot_tsl, merge_rot_tsl, calc_dihedral

class RNA:
    def __init__(self, pdb_path):

        # Load the PDB and store the loaded structure of the RNA
        parser = PDBParser(QUIET=True)
        self.structure = parser.get_structure('struct', pdb_path)

        self.load_coords()
        self.make_main_frames()
        self.make_torsion_angles()

        # print(f"All coords from PDB: {self.coords}")


    def load_coords(self):

        for model in self.structure:
            self.coords_list_of_residues = list()
            for chain in model:
                for residue in chain:

                    residue_coords = dict()
                    residue_coords["resname"] = residue.get_resname()

                    for atom in residue:
                        residue_coords[atom.get_fullname().strip()] = torch.tensor(atom.get_coord())
                    
                    self.coords_list_of_residues.append(residue_coords)
            
        self.coords = torch.zeros((len(self.coords_list_of_residues), len(RNA_CONSTANTS.ATOM_NAMES_PER_RESD["G"]), 3))

        # saving all the coords in a single tensor
        for res_idx, residue_coords_dict in enumerate(self.coords_list_of_residues):
            atom_order = RNA_CONSTANTS.ATOM_NAMES_PER_RESD[residue_coords_dict["resname"]]
            res_coords = list()
            for each_atom in atom_order:
                res_coords.append(residue_coords_dict[each_atom])
            res_coords = torch.stack(res_coords, dim=0)

            self.coords[res_idx,:res_coords.shape[0],:] = res_coords


        
    def make_main_frames(self):

        rots = list()
        tsls = list()

        for each_residue in self.coords_list_of_residues:
            res_name = each_residue["resname"]

            c4_dash, c1_dash, n1_n9 = RNA_CONSTANTS.ATOM_NAMES_PER_RESD[res_name][:3]   # the main frame is made up of C4', C1', N1/N9 based on the residue
            rot, tsl = calc_rot_tsl(each_residue[c4_dash], each_residue[c1_dash], each_residue[n1_n9])

            rots.append(rot)
            tsls.append(tsl)

            # full_frame = torch.cat([rot, tsl.unsqueeze(0)], dim=0)  # the entire frame is a transformation matrix of size 4x3
        
        rots = torch.stack(rots, dim=0)
        #print(f"5 calculated rotation matrices: {rots[:5]}")
        rots = Rotation(rot_mats=rots, quats=None)
        tsls = torch.stack(tsls, dim=0)

        self.main_frames = Rigid(rots, tsls).to_tensor_7().unsqueeze(0)

    def make_torsion_angles(self):

        all_res_torsions = list()

        for each_residue in self.coords_list_of_residues:
            res_name = each_residue["resname"]

            # each_res_torsions = [torch.tensor([torch.cos(torch.tensor([0])), torch.sin(torch.tensor([0]))]),
            #                         torch.tensor([torch.cos(torch.tensor([0])), torch.sin(torch.tensor([0]))])]
            each_res_torsions = list()

            angle_defs = RNA_CONSTANTS.ANGL_INFOS_PER_RESD[res_name]
            for angle_name, _, atom_names in angle_defs:
                torsion_angle = calc_dihedral(each_residue[atom_names[0]], each_residue[atom_names[1]], each_residue[atom_names[2]], each_residue[atom_names[3]])
                each_res_torsions.append(torch.tensor([torch.cos(torsion_angle), torch.sin(torsion_angle)]))

            each_res_torsions = torch.stack(each_res_torsions, dim=0)
            all_res_torsions.append(each_res_torsions)

        all_res_torsions = torch.stack(all_res_torsions, dim=0)
        self.torsion_angles = all_res_torsions


loaded_rna = RNA("rebuilt_pdb.pdb")

from builder_rna_cords import RNAConverterNoOmega
rnaconverter = RNAConverterNoOmega()

parser = PDBParser(QUIET=True)
structure = parser.get_structure('struct', "rebuilt_pdb.pdb")
for model in structure:
    for chain in model:
        seq = []
        for residue in chain:
            seq.append(residue.resname)
coords, mask = rnaconverter.build_cords(''.join(seq), loaded_rna.main_frames, loaded_rna.torsion_angles, rtn_cmsk=True)

print(f"Error of rebuilt coordinates: {torch.sum(torch.abs(loaded_rna.coords - coords))}")

# Import libraries
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

fig = plt.figure(figsize = (10, 7))

# Creating plot
ax = fig.add_subplot(1, 1, 1, projection='3d')

sidechain_points = list()
backbone_points = list()

num_residues = 100
ax.scatter3D(coords[:num_residues,:,0].reshape(-1), coords[:num_residues,:,1].reshape(-1), coords[:num_residues,:,2].reshape(-1), color="black")
ax.scatter3D(loaded_rna.coords[:num_residues,:,0].reshape(-1), loaded_rna.coords[:num_residues,:,1].reshape(-1), loaded_rna.coords[:num_residues,:,2].reshape(-1), color="red")
 
# show plot
plt.show()

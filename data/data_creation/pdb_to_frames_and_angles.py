import numpy as np
import torch
from Bio.PDB import PDBParser

# Import libraries
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

all_outputs = np.load("outputs.npy", allow_pickle=True).item()
for each_key in all_outputs:
    pass
    # print(f"{each_key}: {all_outputs[each_key].shape}")

all_frames = all_outputs['frames'][-1]
all_angles = all_outputs['angles'][-1]

# print(f"Frames: {all_frames}")
# print(f"Angles: {all_angles}")

print(f"Frames: {all_frames.shape}")
print(f"Angles: {all_angles.shape}")

parser = PDBParser(QUIET=True)
structure = parser.get_structure('struct', "unrelaxed_model.pdb")    

# iterate each model, chain, and residue
# printing out the sequence for each chain

for model in structure:
    for chain in model:
        seq = []
        for residue in chain:
            seq.append(residue.resname)

from converter import RNAConverter

rnaconverter = RNAConverter()


cords, mask = rnaconverter.build_cords(''.join(seq), all_frames, all_angles, rtn_cmsk=True)
print(f"Shape of cords from RNAConverter: {cords.shape}")


from RhoFold.rhofold.utils.rigid_utils import Rigid, calc_rot_tsl, calc_angl_rot_tsl, merge_rot_tsl

rigid = Rigid.from_tensor_7(all_frames, normalize_quats=True)
fram = rigid.to_tensor_4x4()
rot = fram[:,:,:3,:3]
tsl = fram[:,:,:3,3:].permute(0,1,3,2)
fram = torch.cat([rot, tsl], dim=2)[:,:,:4,:3].permute(1,0,2,3)

print(f"Fram shape: {fram.shape}")

res_index = 2
frame_res_ind_rot, frame_res_ind_tsl = fram[res_index, 0, :3], fram[res_index, 0, 3]

print(f"Frame Rot shape: {frame_res_ind_rot.shape}, Frame Tsl shape: {frame_res_ind_tsl.shape}")
print(f"Frame Rot: {frame_res_ind_rot}, Frame Tsl: {frame_res_ind_tsl}")
# print(f"Coords: {cords[res_index]}")
print(f"calc_rot_tsl: {calc_rot_tsl(cords[res_index][0], cords[res_index][1], cords[res_index][2])}")

print(f"Closeness Rot: {torch.sum(torch.abs(frame_res_ind_rot - calc_rot_tsl(cords[res_index][0], cords[res_index][1], cords[res_index][2])[0])).item()}")
print(f"Closeness Tsl: {torch.sum(torch.abs(frame_res_ind_tsl - calc_rot_tsl(cords[res_index][0], cords[res_index][1], cords[res_index][2])[1])).item()}")


#### -------------- Figuring out the angles -------------


angl = all_angles.squeeze(dim=0) / (torch.norm(all_angles.squeeze(dim=0), dim=2, keepdim=True) + 0.00000000001)
print(f"Angl shape: {angl.shape}")

res_index = 1
chosen_res_angles = angl[res_index]
print(f"Chosen residue angles: {chosen_res_angles}") 


def calc_dihedral(a, b, c, d):
    """Praxeolitic formula
    1 sqrt, 1 cross product"""

    b0 = -1.0*(b - a)
    b1 = c - b
    b2 = d - c

    # normalize b1 so that it does not influence magnitude of vector
    # rejections that come next
    b1 /= torch.norm(b1)

    # vector rejections
    # v = projection of b0 onto plane perpendicular to b1
    #   = b0 minus component that aligns with b1
    # w = projection of b2 onto plane perpendicular to b1
    #   = b2 minus component that aligns with b1
    v = b0 - torch.dot(b0, b1)*b1
    w = b2 - torch.dot(b2, b1)*b1

    # angle between v and w in a plane is the torsion angle
    # v and w may not be normalized but that's fine since tan is y/x
    x = torch.dot(v, w)
    y = torch.dot(torch.cross(b1, v), w)
    return torch.arctan2(y, x)

def calc_angle(a, b, c):

    ba = a - b
    bc = c - b

    cosine_angle = torch.dot(ba, bc) / (torch.linalg.norm(ba) * torch.linalg.norm(bc))
    angle = torch.arccos(cosine_angle)

    return angle


g_atoms = ["C4'", "C1'", "N9", "C4", "N1", "N2", "N3", "C2", "C5", "C6", "N7", "C8", "O6", "C5'", "C2'", "C3'", "O4'", "O2'", "O3'", "O5'", "P", "OP1", "OP2"]
#           0       1      2    3     4     5     6      7    8      9    10    11     12    13     14     15     16     17    18     19     20    21     22
from tqdm import tqdm

atom_i_index, atom_ii_index, atom_iii_index, atom_iv_index = (18, 15, 0, 13) # index in atom_coordinates
print(torch.cos(calc_dihedral(cords[res_index][atom_i_index], cords[res_index][atom_ii_index], cords[res_index][atom_iii_index], cords[res_index][atom_iv_index])),
        torch.sin(calc_dihedral(cords[res_index][atom_i_index], cords[res_index][atom_ii_index], cords[res_index][atom_iii_index], cords[res_index][atom_iv_index])))
atom_i_index, atom_ii_index, atom_iii_index, atom_iv_index = (0, 1, 2, 3) # index in atom_coordinates
print(torch.cos(calc_dihedral(cords[res_index][atom_i_index], cords[res_index][atom_ii_index], cords[res_index][atom_iii_index], cords[res_index][atom_iv_index])),
        torch.sin(calc_dihedral(cords[res_index][atom_i_index], cords[res_index][atom_ii_index], cords[res_index][atom_iii_index], cords[res_index][atom_iv_index])))

atom_i_index, atom_ii_index, atom_iii_index, atom_iv_index = (0, 1, 2, 3) # index in atom_coordinates
print(torch.cos(calc_dihedral(cords[res_index][atom_i_index], cords[res_index][atom_ii_index], cords[res_index][atom_iii_index], cords[res_index][atom_iv_index])),
        torch.sin(calc_dihedral(cords[res_index][atom_i_index], cords[res_index][atom_ii_index], cords[res_index][atom_iii_index], cords[res_index][atom_iv_index])))
atom_i_index, atom_ii_index, atom_iii_index, atom_iv_index = (2, 1, 0, 13) # index in atom_coordinates
print(torch.cos(calc_dihedral(cords[res_index][atom_i_index], cords[res_index][atom_ii_index], cords[res_index][atom_iii_index], cords[res_index][atom_iv_index])),
        torch.sin(calc_dihedral(cords[res_index][atom_i_index], cords[res_index][atom_ii_index], cords[res_index][atom_iii_index], cords[res_index][atom_iv_index])))
atom_i_index, atom_ii_index, atom_iii_index, atom_iv_index = (1, 0, 13, 19) # index in atom_coordinates
print(torch.cos(calc_dihedral(cords[res_index][atom_i_index], cords[res_index][atom_ii_index], cords[res_index][atom_iii_index], cords[res_index][atom_iv_index])),
        torch.sin(calc_dihedral(cords[res_index][atom_i_index], cords[res_index][atom_ii_index], cords[res_index][atom_iii_index], cords[res_index][atom_iv_index])))
atom_i_index, atom_ii_index, atom_iii_index, atom_iv_index = (0, 13, 19, 20) # index in atom_coordinates
print(torch.cos(calc_dihedral(cords[res_index][atom_i_index], cords[res_index][atom_ii_index], cords[res_index][atom_iii_index], cords[res_index][atom_iv_index])),
        torch.sin(calc_dihedral(cords[res_index][atom_i_index], cords[res_index][atom_ii_index], cords[res_index][atom_iii_index], cords[res_index][atom_iv_index])))


# coeff_sets = list()
# for atom_i_coeff in [0, 1, 2, -1, -2]:
#     for atom_ii_coeff in [0, 1, 2, -1, -2]:
#         for atom_iii_coeff in [0, 1, 2, -1, -2]:
#             coeff_sets.append((atom_i_coeff, atom_ii_coeff, atom_iii_coeff))
            

# from multiprocessing import Pool
# def iterate_i_atom_coeff_set(atom_i_coeff_set):
#     good_sets = list()
#     for atom_ii_coeff_set in coeff_sets:
#         for atom_iii_coeff_set in coeff_sets:

#             atom_i_index, atom_ii_index, atom_iii_index = (1, 2, 3) # index in atom_coordinates
#             calculated_angle = new_angle(atom_i_coeff_set[0] * cords[res_index][atom_i_index] + atom_i_coeff_set[1] * cords[res_index][atom_ii_index] + atom_i_coeff_set[2] * cords[res_index][atom_iii_index],
#                                             atom_ii_coeff_set[0] * cords[res_index][atom_i_index] + atom_ii_coeff_set[1] * cords[res_index][atom_ii_index] + atom_ii_coeff_set[2] * cords[res_index][atom_iii_index],
#                                             atom_iii_coeff_set[0] * cords[res_index][atom_i_index] + atom_iii_coeff_set[1] * cords[res_index][atom_ii_index] + atom_iii_coeff_set[2] * cords[res_index][atom_iii_index],)

#             calculated_angle_cos_sin = torch.tensor([torch.cos(calculated_angle), torch.sin(calculated_angle)])
#             calculated_angle_cos_sin = calculated_angle_cos_sin / (torch.norm(calculated_angle_cos_sin, dim=0, keepdim=True) + 0.00000000001)
            
#             closeness = torch.sum(torch.abs(calculated_angle_cos_sin - chosen_res_angles[2])).item()

#             if closeness < 0.001:
#                 print(f"Closeness: {closeness}, Calculated angles: {calculated_angle_cos_sin}, Reference Angles: {chosen_res_angles[2]}\n Sets: {atom_i_coeff_set} {atom_ii_coeff_set} {atom_iii_coeff_set}")
#                 good_sets.append((atom_i_coeff_set, atom_ii_coeff_set, atom_iii_coeff_set))
    
#     return good_sets

# with Pool(8) as p:
#     print([b for a in p.map(iterate_i_atom_coeff_set, coeff_sets) for b in a])

# exit(1)



fig = plt.figure(figsize = (10, 7))

# Creating plot
ax = fig.add_subplot(1, 1, 1, projection='3d')

sidechain_points = list()
backbone_points = list()

num_residues = 50000
ax.scatter3D(cords[:num_residues,:,0].reshape(-1), cords[:num_residues,:,1].reshape(-1), cords[:num_residues,:,2].reshape(-1), color="black")
 
# show plot
plt.show()





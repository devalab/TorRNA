"""
This code generates the coordinates of a single RNA residue and also has a slider to figure out what the "angl_0" variable is in the RNAConverter code
"""


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


import os
import torch
import numpy as np
import logging
from collections import defaultdict
from RhoFold.rhofold.utils.constants import RNA_CONSTANTS
from RhoFold.rhofold.utils.rigid_utils import Rigid, calc_rot_tsl, calc_angl_rot_tsl, merge_rot_tsl

class RNAConverter():
    """RNA Structure Converter."""

    def __init__(self):
        """"""

        self.eps = 1e-4
        self.__init()

    def __init(self):
        """"""

        self.cord_dict = defaultdict(dict) # this dictionary will contain the coordinates of atoms in each residue from RNA_CONSTANTS.ATOM_INFOS_PER_RESD
        for resd_name in RNA_CONSTANTS.RESD_NAMES: # A, G, U, C
            for atom_name, _, cord_vals in RNA_CONSTANTS.ATOM_INFOS_PER_RESD[resd_name]:
                self.cord_dict[resd_name][atom_name] = torch.tensor(cord_vals, dtype=torch.float32)

        trans_dict_all = {}
        for resd_name in RNA_CONSTANTS.RESD_NAMES: # A, G, U, C
            trans_dict = {}
            cord_dict = {}

            atom_infos = RNA_CONSTANTS.ATOM_INFOS_PER_RESD[resd_name] # atom_infos has a list of all atoms in the residue, their group, and coordinates
            angl_infos = RNA_CONSTANTS.ANGL_INFOS_PER_RESD[resd_name] # angl_infos has a list of the list of angles_{0,1,2,3}, a boolean (False), and the list of atoms that make the angle
            n_angls = len(angl_infos)
            
            for atom_name, idx_rgrp, _ in atom_infos:
                if idx_rgrp == 0:   # get the coords of just the main backbone (group id 0, C4', C1', N1)
                    cord_dict[atom_name] = self.cord_dict[resd_name][atom_name] # update the coords of the main backbone atoms

            # transformation=rotation+translation to omega from the main frame, to phi from the main frame
            # they are initialized to be the identity transformation
            trans_dict['omega-main'] = (torch.eye(3, dtype=torch.float32), torch.zeros((3), dtype=torch.float32))
            trans_dict['phi-main'] = (torch.eye(3, dtype=torch.float32), torch.zeros((3), dtype=torch.float32))

            for idx_angl, (angl_name, _, atom_names_sel) in enumerate(angl_infos):
                # x1, x2, x3 has the reference coordinates of the first 3 atoms that make the angle_{0,1,2,3}
                x1 = cord_dict[atom_names_sel[0]]
                x2 = cord_dict[atom_names_sel[1]]
                x3 = cord_dict[atom_names_sel[2]]
                rot, tsl_vec = calc_rot_tsl(x1, x3, x3 + (x3 - x2))
                trans_dict['%s-main' % angl_name] = (rot, tsl_vec)  # transformations to the angles_{0,1,2,3} from the main frame

                for atom_name, idx_rgrp, _ in atom_infos:
                    if idx_rgrp == idx_angl + 3:
                        cord_dict[atom_name] = tsl_vec + torch.sum( # building the coordinates of the individual atoms
                            rot * self.cord_dict[resd_name][atom_name].view(1, 3), dim=1)

            # calculating the transformations between angles_{0,1,2,3}
            for idx_angl_src in range(1, n_angls - 1):
                idx_angl_dst = idx_angl_src + 1
                angl_name_src = angl_infos[idx_angl_src][0]
                angl_name_dst = angl_infos[idx_angl_dst][0]
                rot_src, tsl_vec_src = trans_dict['%s-main' % angl_name_src]
                rot_dst, tsl_vec_dst = trans_dict['%s-main' % angl_name_dst]
                rot = torch.matmul(rot_src.transpose(1, 0), rot_dst)
                tsl_vec = torch.matmul(rot_src.transpose(1, 0), tsl_vec_dst - tsl_vec_src)
                trans_dict['%s-%s' % (angl_name_dst, angl_name_src)] = (rot, tsl_vec)

            # trans_dict_all has the transformations between each set of angles, and the angles to the main frame for all residues
            trans_dict_all[resd_name] = trans_dict

        self.trans_dict_init = trans_dict_all

    def build_cords(self, seq, fram, angl, phi_radian, rtn_cmsk=False):

        # initialization
        n_resds = 1
        device = angl.device

        # angl is the pair (cos, sin) of the angle given, this normalizes it to make sure cos^2 + sin^2 = 1 for the angle
        angl = angl.squeeze(dim=0) / (torch.norm(angl.squeeze(dim=0), dim=2, keepdim=True) + self.eps)
        rigid = Rigid.from_tensor_7(fram, normalize_quats=True)
        fram = rigid.to_tensor_4x4()
        rot = fram[:,:,:3,:3]
        tsl = fram[:,:,:3,3:].permute(0,1,3,2)

        fram = torch.cat([rot, tsl], dim=2)[:,:,:4,:3].permute(1,0,2,3)
        fmsk = torch.ones((n_resds, 1), dtype=torch.int8, device=device)
        amsk = torch.ones((n_resds, RNA_CONSTANTS.N_ANGLS_PER_RESD_MAX), dtype=torch.int8, device=device)
        cord = torch.zeros((n_resds, RNA_CONSTANTS.ATOM_NUM_MAX, 3), dtype=torch.float32, device=device)
        cmsk = torch.zeros((n_resds, RNA_CONSTANTS.ATOM_NUM_MAX), dtype=torch.int8, device=device)

        for resd_name in RNA_CONSTANTS.RESD_NAMES:
            idxs = [x for x in range(n_resds) if seq[x] == resd_name]
            if len(idxs) == 0:
                continue
            cord[idxs], cmsk[idxs] =\
                self.__build_cord(resd_name, fram[idxs], fmsk[idxs], angl[idxs], phi_radian, amsk[idxs])

        return (cord, cmsk) if rtn_cmsk else (cord)

    def __build_cord(self, resd_name, fram, fmsk, angl, phi_radian, amsk):
        """"""

        # initialization
        device = fram.device
        n_resds = fram.shape[0]
        atom_names_all = RNA_CONSTANTS.ATOM_NAMES_PER_RESD[resd_name]
        atom_names_pad = atom_names_all + ['X'] * (RNA_CONSTANTS.ATOM_NUM_MAX - len(atom_names_all))
        atom_infos_all = RNA_CONSTANTS.ATOM_INFOS_PER_RESD[resd_name]

        cord_dict = defaultdict(
            lambda: torch.zeros((n_resds, 3), dtype=torch.float32, device=device))
        cmsk_vec_dict = defaultdict(lambda: torch.zeros((n_resds), dtype=torch.int8, device=device))

        fram_null = torch.tensor(
            [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=torch.float32, device=device)
        fram_dict = defaultdict(lambda: fram_null.unsqueeze(dim=0).repeat(n_resds, 1, 1))
        fmsk_vec_dict = defaultdict(lambda: torch.zeros((n_resds), dtype=torch.int8, device=device))

        trans_dict = {'main': (fram[:, 0, :3], fram[:, 0, 3])}
        rot_curr, tsl_curr = trans_dict['main']

        # Calculate the coordinates of the the atoms with the "group number" = 0 in ATOM_INFOS_PER_RESD (C4', C1', N9/N1). This is the main/base frame of the backbone of each residue
        atom_names_sel = [x[0] for x in atom_infos_all if x[1] == 0]
        for atom_name_sel in atom_names_sel:
            cord_vec = self.cord_dict[resd_name][atom_name_sel].to(device)
            cord_dict[atom_name_sel] = \
                tsl_curr + torch.sum(rot_curr * cord_vec.view(1, 1, 3), dim=2)
            cmsk_vec_dict[atom_name_sel] = fmsk[:, 0]

        # determine 3D coordinates of atoms belonging to side-chain rigid-groups
        angl_infos_all = RNA_CONSTANTS.ANGL_INFOS_PER_RESD[resd_name]
        rgrp_names_all = ['omega', 'phi'] + [x[0] for x in angl_infos_all]

        for idx_rgrp, rgrp_name_curr in enumerate(rgrp_names_all):
            # rgrp_name_curr - ['omega', 'phi', 'angl_0', 'angl_1', 'angl_2', 'angl_3']
            # idx_rgrp       -     0       1        2         3         4         5

            if rgrp_name_curr in ['omega', 'phi', 'angl_0', 'angl_1']:  # all these rigid-groups have their orientations w.r.t. the main frame
                rgrp_name_prev = 'main'
            else:   # angl_2 and angl_3 have their orientations w.r.t angl_1 and angl_2 respectively
                rgrp_name_prev = 'angl_%d' % (int(rgrp_name_curr[-1]) - 1)
            
            rot_prev, tsl_prev = trans_dict[rgrp_name_prev] # get the rotation and translation from universal ground frame to the "previous" rigid group frame
            rot_base, tsl_vec_base = \
                self.trans_dict_init[resd_name]['%s-%s' % (rgrp_name_curr, rgrp_name_prev)] # get the rotation and translation from the "previous" rigid group frame to the "current" rigid group frame
            rot_base = rot_base.unsqueeze(dim=0).to(device)
            tsl_base = tsl_vec_base.unsqueeze(dim=0).to(device)
            
            if idx_rgrp == 1:
                rot_addi, tsl_addi = calc_angl_rot_tsl(torch.tensor([[torch.cos(phi_radian).item(), torch.sin(phi_radian).item()]]))   # calculate the additional rotation and translation based on the input angle of the torsional angle of the "current" rigid group frame TODO UNDERSTAND
            else:
                rot_addi, tsl_addi = calc_angl_rot_tsl(angl[:, idx_rgrp])   # calculate the additional rotation and translation based on the input angle of the torsional angle of the "current" rigid group frame TODO UNDERSTAND
            rot_curr, tsl_curr = merge_rot_tsl(
                rot_prev, tsl_prev, rot_base, tsl_base, rot_addi, tsl_addi)
            trans_dict[rgrp_name_curr] = (rot_curr, tsl_curr)   # save the rotation and translation from universal ground frame to the "current" rigid group frame

            fram_dict[rgrp_name_curr] = \
                torch.cat([rot_curr, tsl_curr.unsqueeze(dim=1)], dim=1)
            fmsk_vec_dict[rgrp_name_curr] = fmsk[:, 0] * amsk[:, idx_rgrp]

            atom_names_sel = [x[0] for x in atom_infos_all if x[1] == idx_rgrp + 1]
            for atom_name_sel in atom_names_sel:
                cord_vec = self.cord_dict[resd_name][atom_name_sel].to(device)

                cord_dict[atom_name_sel] = \
                    tsl_curr + torch.sum(rot_curr * cord_vec.view(1, 1, 3), dim=2)
                cmsk_vec_dict[atom_name_sel] = fmsk_vec_dict[rgrp_name_curr]

        cmsk = torch.stack([cmsk_vec_dict[x] for x in atom_names_pad][:RNA_CONSTANTS.ATOM_NUM_MAX], dim=1)
        cord = torch.stack([cord_dict[x] for x in atom_names_pad][:RNA_CONSTANTS.ATOM_NUM_MAX], dim=1)

        return cord, cmsk



rnaconverter = RNAConverter()

angl = all_angles.squeeze(dim=0) / (torch.norm(all_angles.squeeze(dim=0), dim=2, keepdim=True) + 0.00000000001)



import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button


fig = plt.figure(figsize = (10, 7))

# Creating plot
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.set_title("What does this variable do in a complicated piece of code?")


init_radian = torch.acos(angl[0][1][0])

# adjust the main plot to make room for the sliders
fig.subplots_adjust(left=0.25, bottom=0.25)

# Make a horizontal slider to control the frequency.
axrad = fig.add_axes([0.25, 0.1, 0.65, 0.03])
radian_slider = Slider(
    ax=axrad,
    label='Angle 0 (Radian)',
    valmin=0,
    valmax=7,
    valinit=init_radian,
)


# The function to be called anytime a slider's value changes
def update(val):

    radian_val = torch.tensor(radian_slider.val)
    cords = rnaconverter.build_cords(''.join(seq), all_frames[:1], all_angles[:1], radian_val, rtn_cmsk=True)[0]

    ax.cla()
    ax.scatter3D(cords[:,:,0].reshape(-1), cords[:,:,1].reshape(-1), cords[:,:,2].reshape(-1), color="red")


    g_atoms = ["C4'", "C1'", "N9", "C4", "N1", "N2", "N3", "C2", "C5", "C6", "N7", "C8", "O6", "C5'", "C2'", "C3'", "O4'", "O2'", "O3'", "O5'", "P", "OP1", "OP2"]

    for i in range(len(cords[:,:,0].reshape(-1))): #plot each point + it's index as text above
        ax.text(cords[:,:,0].reshape(-1)[i], cords[:,:,1].reshape(-1)[i], cords[:,:,2].reshape(-1)[i],  '%s' % (str(g_atoms[i])), size=10, zorder=1, color='k') 

    ax.set_title("What does this variable do in a complicated piece of code?")

    ax.set_xlim(-20,-10)
    ax.set_ylim(-23.5,-14)
    ax.set_zlim(-3, 4)

    fig.canvas.draw_idle()


# register the update function with each slider
radian_slider.on_changed(update)
update(0)
plt.show()

# cords, mask = rnaconverter.build_cords(''.join(seq), all_frames[:1], all_angles[:1], radian, rtn_cmsk=True)
# print(f"Shape of cords from RNAConverter: {cords.shape}")


# fig = plt.figure(figsize = (10, 7))

# # Creating plot
# ax = fig.add_subplot(1, 1, 1, projection='3d')

# sidechain_points = list()
# backbone_points = list()

# ax.scatter3D(cords[:,:,0].reshape(-1), cords[:,:,1].reshape(-1), cords[:,:,2].reshape(-1), color="black")

# # show plot
# plt.show()





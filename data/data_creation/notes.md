Rhofold/model/structure_module.py - StructureModule()
    - outputs['frames'][-1] and outputs['angles'][-1] are used to build the entire RNA coordinates

Rhofold/utils/converter.py - RNAConverter()
    - build_cords - Uses Frames and Angles from Structure Module
        - __build_cord

The "frames" are calculated as the rotation and translation to move into the frame of reference of a residue's angle made by C4', C1', N1/N9
    - Calculate the frame's rotation and translation with `calc_rot_tsl(C4' cords, C1' cords, N1/N9 cords)`

The "angles" are calculated as the (cos, sin) of the dihedral angles as defined in ANGL_INFOS_PER_RESD.
    - Calculate the angle as `new_dihedral(p1, p2, p3, p4)`




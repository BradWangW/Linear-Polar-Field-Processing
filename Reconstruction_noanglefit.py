import numpy as np
import polyscope as ps
from Functions.Auxiliary import *
from Functions.Mesh_noanglefit import Triangle_mesh
import os
import polyscope.imgui as psim


# Increase the recursion limit
np.set_printoptions(threshold=np.inf)


if __name__ == '__main__':
    
    V, F = load_off_file(os.path.join('..', 'data', 'spherers.off'))
    E = obtain_E(F)

    mesh = Triangle_mesh(V, F)
    
    singularity_info = np.array([
        [1000, 1],
        [900, 1],
        # [2000, -1],
        # [3000, -1],
        # [2500, -1],
        # [1500, -1]
    ])
    F_singular = singularity_info[:, 0]
    indices = singularity_info[:, 1]
    
    mesh.dynamic_field(F_singular, indices)
    # mesh.dynamic_field_along_curve(F_singular, indices, 20)
            

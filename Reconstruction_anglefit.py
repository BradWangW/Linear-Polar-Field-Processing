import numpy as np
import polyscope as ps
from Functions.Auxiliary import *
from Functions.Mesh_anglefit import Triangle_mesh
import os
import polyscope.imgui as psim


# Increase the recursion limit
np.set_printoptions(threshold=np.inf)


if __name__ == '__main__':
    
    V, F = load_off_file(os.path.join('..', 'data', 'Kitten.off'))
    E = obtain_E(F)

    mesh = Triangle_mesh(V, F)
    
    singularity_info = np.array([
        [100, -1],
        [900, 1]
    ])
    F_singular = singularity_info[:, 0]
    indices = singularity_info[:, 1]
    
    mesh.dynamic_field(F_singular, indices)
            

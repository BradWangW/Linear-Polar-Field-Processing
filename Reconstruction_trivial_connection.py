import numpy as np
import polyscope as ps
from Functions.Auxiliary import *
from Functions.Mesh_trivial_connection import Triangle_mesh
import os
import polyscope.imgui as psim


# Increase the recursion limit
np.set_printoptions(threshold=np.inf)

if __name__ == '__main__':
    
    # V, F = load_off_file(os.path.join('..', 'data', 'torus.off'))
    # V, F = load_off_file(os.path.join('..', 'data', 'spherers.off'))
    # V, F = load_off_file(os.path.join('..', 'data', 'Kitten.off'))
    # V, F = load_off_file(os.path.join('..', 'data', 'rocker-arm2500.off'))
    
    
    # V, F = load_off_file(os.path.join('..', 'data_patho', 'Chassis_-_upright_f_l-1_front_upright-1_B9-1.off'))
    # V, F = load_off_file(os.path.join('..', 'data_patho', 'rocker-arm1250.off'))
    # V, F = load_off_file(os.path.join('..', 'data_patho', 'cheburashka-subdivision.off'))
    V, F = load_off_file(os.path.join('..', 'data_patho', 'fertility.off'))
    
    E = obtain_E(F)

    mesh = Triangle_mesh(V, F)
    
    singularity_info = np.array([
        [0, -1],
        [200, -1],
        [300, -1],
        [700, -1],
        [2500, -1],
        [1500, -1]
    ])
    V_singular = singularity_info[:, 0]
    indices = singularity_info[:, 1]
    
    mesh.dynamic_field_vertex(V_singular, indices, interval_sample=0.5)
    # mesh.dynamic_field_along_curve(F_singular, indices, 20)
            

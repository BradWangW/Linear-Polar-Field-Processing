import numpy as np
import scipy.linalg
import polyscope as ps
from scipy.sparse import lil_matrix, csr_matrix, csc_matrix, coo_matrix, linalg, bmat, diags, vstack
from scipy.sparse.linalg import spsolve, splu, lsqr
from tqdm import tqdm
import os
from contextlib import redirect_stdout, redirect_stderr
from DAFunctions import load_off_file, compute_areas_normals, compute_laplacian, compute_mean_curvature_normal, compute_edge_list, compute_angle_defect
from scipy.optimize import minimize
from Auxiliary import *
from Mesh import Triangle_mesh


# Increase the recursion limit
np.set_printoptions(threshold=np.inf)


if __name__ == '__main__':
    
    
    # A minimal triangulated tetrahedron
    V = np.array([
        [1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]
    ], dtype=float)
    F = np.array([
        [0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]
    ])
    singularities = np.array([[1/3, 1/3, -1/3], [-1, -1, 1]])
    indices = [1, 1]
    # singularities = np.array([[1/3, 1/3, -1/3]])
    # indices = [2]
    v_init = 10
    z_init = -1

    mesh = Triangle_mesh(V, F)

    coeffs_truth = np.zeros((len(F), 1, 2), dtype=complex)

    # Truth coefficient for the linear term
    coeffs_truth[:, 0, 0] = 1

    zeros_per_face = np.array([
        [1/3, 1/3, -1/3], 
        (2 * (V[0] + V[2]) - V[3])/2, 
        (2 * (V[0] + V[1]) - V[3])/2, 
        (2 * (V[1] + V[2]) - V[3])/2, 
    ])

    Z_zero_per_face = complex_projection(
        mesh.B1, mesh.B2, mesh.normals, zeros_per_face, diagonal=True
    )

    coeffs_truth[:, 0, 1] = -coeffs_truth[:, 0, 0] * Z_zero_per_face

    # Compute the fields
    field_truth, field = mesh.vector_field_from_truth(coeffs_truth, singularities, indices)

    V_truth = V + 3

    points_truth, vectors_truth = sample_points_and_vectors(V, F, field_truth, num_samples=25)
    points, vectors = sample_points_and_vectors(V, F, field, num_samples=25)

    print(points.shape, vectors.shape, points_truth.shape, vectors_truth.shape)

    # normalise the vectors
    vectors_truth = vectors_truth / np.linalg.norm(vectors_truth, axis=1)[:, None]
    vectors = vectors / np.linalg.norm(vectors, axis=1)[:, None]

    ps.init()
    ps_mesh_truth = ps.register_surface_mesh("Truth Mesh", V_truth, F)
    ps_mesh = ps.register_surface_mesh("Input Mesh", V, F)

    ps_field_truth = ps.register_point_cloud("Centroids", points_truth, enabled=True)
    ps_field_truth.add_vector_quantity('Samples', vectors_truth, enabled=True)

    ps_field = ps.register_point_cloud("Centroids", points, enabled=True)
    ps_field.add_vector_quantity('Samples', vectors, enabled=True)
    
    ps.register_point_cloud("singularity marker", singularities, enabled=True)

    ps.show()
            

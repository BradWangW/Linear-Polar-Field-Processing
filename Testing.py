import numpy as np
import polyscope as ps
from Functions.Auxiliary import *
from Functions.Mesh import Triangle_mesh
import os


# Increase the recursion limit
np.set_printoptions(threshold=np.inf)


if __name__ == '__main__':
    
    V, F = load_off_file(os.path.join('..', 'data', 'spherers.off'))
    singularities = np.array([
        0.3 * V[F[0, 0]] + 0.3 * V[F[0, 1]] + 0.4 * V[F[0, 2]],
        V[F[100, 0]]
    ])
    indices = [1, 1]
    v_init = 100
    z_init = 1
    
    # # A minimal triangulated tetrahedron
    # V = np.array([
    #     [1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]
    # ], dtype=float)
    # F = np.array([
    #     [0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]
    # ])
    # singularities = np.array([[1/3, 1/3, -1/3], [-1, -1, 1]])
    # # singularities = np.array([[1, 1, 1], [-1/3, -1/3, -1/3]])
    # # singularities = np.array([[1/3, 1/3, -1/3]])
    # # singularities = np.array([[1/3, 1/3, -1/3], [-1/3, -1/3, -1/3]])
    # indices = [1, 1]
    # v_init = 0
    # z_init = 1

    mesh = Triangle_mesh(V, F)

    
    linear_field = mesh.vector_field(
        singularities, [1, 1], v_init, z_init
    )
    
    linear_conj_field = mesh.vector_field(
        singularities, [1, 1], v_init, z_init, 
        conj=True
    )
    
    posis, vectors = mesh.sample_points_and_vectors(
        linear_field, num_samples=3
    )
    posis_conj, vectors_conj = mesh.sample_points_and_vectors(
        linear_conj_field, num_samples=3
    )
    
    V_truth = V.copy()
    V_truth[:, [0, 2]] += 0.7
    posis_conj[:, [0, 2]] += 0.7
    singularities_truth = singularities.copy()
    singularities_truth[:, [0, 2]] += 0.7
    
    vectors /= np.linalg.norm(vectors, axis=1)[:, None]
    vectors_conj /= np.linalg.norm(vectors_conj, axis=1)[:, None]
    
    ps.init()
    ps_mesh_truth = ps.register_surface_mesh("Conj Mesh", V_truth, F)
    ps_mesh = ps.register_surface_mesh("Input Mesh", V, F)

    ps_field_truth = ps.register_point_cloud("Centroids_conj", posis_conj, enabled=True, radius=0)
    ps_field_truth.add_vector_quantity('Samples_conj', vectors_conj, enabled=True)

    ps_field = ps.register_point_cloud("Centroids", posis, enabled=True, radius=0)
    ps_field.add_vector_quantity('Samples', vectors, enabled=True)
    
    ps.register_point_cloud("singularity marker", singularities, enabled=True)
    ps.register_point_cloud("singularity_conj marker", singularities_truth, enabled=True)

    ps.show()
            

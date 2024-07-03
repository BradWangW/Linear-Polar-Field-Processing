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
    
    # V, F = load_off_file(os.path.join('..', 'data', 'Kitten.off'))
    # print(V.shape, F.shape)
    # singularities = np.array([
    #     V[F[100, 0]],
    #     0.3 * V[F[10, 0]] + 0.3 * V[F[10, 1]] + 0.4 * V[F[10, 2]]
    # ])
    # print(singularities)
    # indices = [1, 1]
    # v_init = 100
    # z_init = 1
    
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
    
    field = mesh.vector_field(
        singularities, indices, v_init, z_init, 
        six_eq_fit_linear=False
    )
    
    posis, vectors = sample_points_and_vectors(V, F, field, num_samples=6)
    
    vectors /= np.linalg.norm(vectors, axis=1)[:, None]

    ps.init()
    ps_mesh = ps.register_surface_mesh("Input Mesh", V, F)

    ps_field = ps.register_point_cloud("Field_sample", posis, enabled=True, radius=0)
    ps_field.add_vector_quantity('Field', vectors, enabled=True)
    
    ps.register_point_cloud("singularity marker", singularities, enabled=True)

    ps.show()
            

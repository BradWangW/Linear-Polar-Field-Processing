import numpy as np
import polyscope as ps
from Functions.Auxiliary import *
from Functions.MeshMassiveSuperpose import Triangle_mesh
import os


# Increase the recursion limit
np.set_printoptions(threshold=np.inf)


if __name__ == '__main__':
    
    # V, F = load_off_file(os.path.join('..', 'data', 'spherers.off'))
    # E = obtain_E(F)
    # # singularities = np.array([
    # #     0.2 * V[F[0, 0]] + 0.2 * V[F[0, 1]] + 0.6 * V[F[0, 2]],
    # #     0.6 * V[F[0, 0]] + 0.2 * V[F[0, 1]] + 0.2 * V[F[0, 2]], 
    # #     0.2 * V[F[100, 0]] + 0.2 * V[F[100, 1]] + 0.6 * V[F[100, 2]],
    # #     0.6 * V[F[100, 0]] + 0.2 * V[F[100, 1]] + 0.2 * V[F[100, 2]]
    # # ])
    # singularities = np.array([
    #     0.2 * V[F[0, 0]] + 0.2 * V[F[0, 1]] + 0.6 * V[F[0, 2]],
    #     0.6 * V[F[5, 0]] + 0.2 * V[F[5, 1]] + 0.2 * V[F[5, 2]], 
    #     0.2 * V[F[100, 0]] + 0.2 * V[F[100, 1]] + 0.6 * V[F[100, 2]],
    #     (V[E[100, 0]] + V[E[100, 1]])/2
    # ])
    # # singularities = np.array([
    # #     0.2 * V[F[0, 0]] + 0.2 * V[F[0, 1]] + 0.6 * V[F[0, 2]],
    # #     0.2 * V[F[10, 0]] + 0.2 * V[F[10, 1]] + 0.6 * V[F[10, 2]]
    # # ])   
    # # singularities = np.array([
    # #     0.2 * V[F[0, 0]] + 0.2 * V[F[0, 1]] + 0.6 * V[F[0, 2]], 
    # #     V[E[100, 0]]
    # # ])
    # indices = [1, 1, -1, 1]
    # v_init = 10
    # z_init = 1
    
    V, F = load_off_file(os.path.join('..', 'data', 'Kitten.off'))
    singularities = np.array([
        0.2 * V[F[0, 0]] + 0.2 * V[F[0, 1]] + 0.6 * V[F[0, 2]],
        0.6 * V[F[1, 0]] + 0.2 * V[F[1, 1]] + 0.2 * V[F[1, 2]], 
        0.2 * V[F[100, 0]] + 0.2 * V[F[100, 1]] + 0.6 * V[F[100, 2]],
        0.6 * V[F[101, 0]] + 0.2 * V[F[101, 1]] + 0.2 * V[F[101, 2]]
    ])
    indices = [1, -1, 1, 1]
    v_init = 10
    z_init = 1j
    
    # V, F = load_off_file(os.path.join('..', 'data', 'cow.off'))
    # singularities = np.array([
    #     0.2 * V[F[0, 0]] + 0.2 * V[F[0, 1]] + 0.6 * V[F[0, 2]],
    #     0.6 * V[F[0, 0]] + 0.2 * V[F[0, 1]] + 0.2 * V[F[0, 2]], 
    #     0.2 * V[F[100, 0]] + 0.2 * V[F[100, 1]] + 0.6 * V[F[100, 2]],
    #     0.6 * V[F[100, 0]] + 0.2 * V[F[100, 1]] + 0.2 * V[F[100, 2]]
    # ])
    # indices = [1, -1, 1, 1]
    # v_init = 10
    # z_init = 1j
    
    # A minimal triangulated tetrahedron
    # V = np.array([
    #     [1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]
    # ], dtype=float)
    # F = np.array([
    #     [0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]
    # ])
    # # singularities = np.array([[1/3, 1/3, -1/3], [-1, -1, 1]])
    # # singularities = np.array([[1, 1, 1], [-1/3, -1/3, -1/3]])
    # # singularities = np.array([[1/3, 1/3, -1/3]])
    # singularities = np.array([
    #     0.333 * V[F[0, 0]] + 0.333 * V[F[0, 1]] + 0.334 * V[F[0, 2]],
    #     V[3]
    # ])
    # indices = [1, 1]
    # v_init = 0
    # z_init = 1

    mesh = Triangle_mesh(V, F)
    
    mesh.initialise_field_processing()
    
    field = mesh.vector_field(
        singularities, indices, v_init, z_init
    )
    
    posis, vectors = mesh.sample_points_and_vectors(
        field, num_samples=3, margin=0.15, singular_detail=True
        )
    
    vectors /= np.linalg.norm(vectors, axis=1)[:, None]

    ps.init()
    ps_mesh = ps.register_surface_mesh("Input Mesh", V, F)

    ps_field = ps.register_point_cloud("Field_sample", posis, enabled=True, radius=0)
    ps_field.add_vector_quantity('Field', vectors, enabled=True)
    
    # for f in mesh.F_singular:
    #     s = mesh.singularities_f[f]
    #     ps.register_point_cloud(f"singularity for face {f}", np.array(s), enabled=True)
    
    ps.register_point_cloud("singularity marker", singularities, enabled=True)

    ps.show()
            

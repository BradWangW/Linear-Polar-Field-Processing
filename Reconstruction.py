import numpy as np
import polyscope as ps
from Functions.Auxiliary import *
from Functions.Mesh import Triangle_mesh
import os


# Increase the recursion limit
np.set_printoptions(threshold=np.inf)


if __name__ == '__main__':
    
    V, F = load_off_file(os.path.join('..', 'data', 'spherers.off'))
    E = obtain_E(F)
    # singularities = np.array([
    #     0.2 * V[F[0, 0]] + 0.2 * V[F[0, 1]] + 0.6 * V[F[0, 2]],
    #     0.6 * V[F[0, 0]] + 0.2 * V[F[0, 1]] + 0.2 * V[F[0, 2]], 
    #     0.2 * V[F[100, 0]] + 0.2 * V[F[100, 1]] + 0.6 * V[F[100, 2]],
    #     0.6 * V[F[100, 0]] + 0.2 * V[F[100, 1]] + 0.2 * V[F[100, 2]]
    # ])     
    singularities = np.array([
        0.2 * V[F[70, 0]] + 0.2 * V[F[70, 1]] + 0.6 * V[F[70, 2]],
        0.6 * V[F[70, 0]] + 0.2 * V[F[70, 1]] + 0.2 * V[F[70, 2]],
        V[F[205, 0]],
        0.2 * V[F[200, 0]] + 0.2 * V[F[200, 1]] + 0.6 * V[F[200, 2]],
        0.2 * V[F[100, 0]] + 0.2 * V[F[100, 1]] + 0.6 * V[F[100, 2]],
        0.6 * V[F[103, 0]] + 0.2 * V[F[103, 1]] + 0.2 * V[F[103, 2]],
        (V[E[100, 0]] + V[E[100, 1]])/2, 
        (V[E[100, 0]] + V[E[100, 1]])/2, 
        0.2 * V[F[0, 0]] + 0.2 * V[F[0, 1]] + 0.6 * V[F[0, 2]],
        0.2 * V[F[0, 0]] + 0.2 * V[F[0, 1]] + 0.6 * V[F[0, 2]],
        0.2 * V[F[0, 0]] + 0.2 * V[F[0, 1]] + 0.6 * V[F[0, 2]],
        0.2 * V[F[0, 0]] + 0.2 * V[F[0, 1]] + 0.6 * V[F[0, 2]]
    ])
    indices = [1, -1, 1, -1, 1, -1, -1, -1, 1, 1, 1, 1]
    v_init = 10
    z_init = 1

    mesh = Triangle_mesh(V, F)
    
    mesh.initialise_field_processing()
    
    field = mesh.vector_field(
        singularities, indices, v_init, z_init
    )
    
    posis, vectors = mesh.sample_points_and_vectors(
        field, num_samples=3, margin=0.15, singular_detail=True, num_samples_detail=15, margin_detail=0.025
        )
    
    vectors /= np.linalg.norm(vectors, axis=1)[:, None]

    ps.init()
    ps_mesh = ps.register_surface_mesh("Input Mesh", V, F)

    ps_field = ps.register_point_cloud("Field_sample", posis, enabled=True, radius=0)
    ps_field.add_vector_quantity('Field', vectors, enabled=True, length=0.0075)
    
    # for f in mesh.F_singular:
    #     ps.register_point_cloud(f"Singularities{f}", np.array(mesh.singularities_f[f]), enabled=True, radius=0.002)
        
    for f in mesh.F_over_pi:
        ps.register_surface_mesh(f"F_over_pi{f}", mesh.V_subdivided[f], mesh.F_subdivided[f], enabled=True)
    
    ps.register_point_cloud("singularity marker", singularities, enabled=True, radius=0.002)

    ps.show()
            

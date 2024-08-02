import numpy as np
import polyscope as ps
from Functions.Auxiliary import *
from Functions.Mesh import Triangle_mesh
import os
import polyscope.imgui as psim


# Increase the recursion limit
np.set_printoptions(threshold=np.inf)


if __name__ == '__main__':
    
    V, F = load_off_file(os.path.join('..', 'data', 'spherers.off'))
    E = obtain_E(F)

    mesh = Triangle_mesh(V, F)
    
    mesh.initialise_field_processing()
    
    singularities = np.array([
        0.2 * V[F[70, 0]] + 0.2 * V[F[70, 1]] + 0.6 * V[F[70, 2]],
        0.6 * V[F[70, 0]] + 0.2 * V[F[70, 1]] + 0.2 * V[F[70, 2]],
        V[F[205, 0]],
        0.2 * V[F[205, 0]] + 0.2 * V[F[205, 1]] + 0.6 * V[F[205, 2]],
        0.2 * V[F[100, 0]] + 0.2 * V[F[100, 1]] + 0.6 * V[F[100, 2]],
        0.6 * V[F[103, 0]] + 0.2 * V[F[103, 1]] + 0.2 * V[F[103, 2]],
        (V[E[100, 0]] + V[E[100, 1]])/2,
        0.2 * V[F[0, 0]] + 0.2 * V[F[0, 1]] + 0.6 * V[F[0, 2]],
        V[F[300, 0]],
        0.2 * V[F[305, 0]] + 0.2 * V[F[305, 1]] + 0.6 * V[F[305, 2]],
        0.2 * V[F[305, 0]] + 0.2 * V[F[305, 1]] + 0.6 * V[F[305, 2]],
        0.2 * V[F[5, 0]] + 0.2 * V[F[5, 1]] + 0.6 * V[F[5, 2]]
    ])
    indices = [-1, 1, -1, 1, 1, 1, -1, -1, -1, 1, 1, 1]
    v_init = 10
    z_init = 1
    
    U = mesh.corner_field(singularities, indices, v_init, z_init)
    
    coeffs, coeffs_singular, coeffs_subdivided = mesh.reconstruct_linear_from_corners(U)
    
    posis, vectors_complex, F_involved = mesh.sample_field(
        coeffs, coeffs_singular, coeffs_subdivided, num_samples=3, margin=0.15, 
        singular_detail=True, num_samples_detail=12, margin_detail=0.05
    )
    
    B1, B2 = mesh.B1[F_involved], mesh.B2[F_involved]
    
    vectors = vectors_complex.real[:, None] * B1 + vectors_complex.imag[:, None] * B2
    
    vectors /= np.linalg.norm(vectors, axis=1)[:, None]

    ps.init()
    ps_mesh = ps.register_surface_mesh("Input Mesh", V, F)

    ps_field = ps.register_point_cloud("Field_sample", posis, enabled=True, radius=0)
    ps_field.add_vector_quantity('Field', vectors, enabled=True, length=0.0075)
            
    for f in mesh.F_over_pi:
        ps.register_surface_mesh(f"F_over_pi{f}", mesh.V_subdivided[f], mesh.F_subdivided[f], enabled=True)
    
    
    ps.register_point_cloud("singularity marker", singularities, enabled=True, radius=0.002)
    
    vectors_complex_unrot = vectors_complex.copy()
    beta = 1
    rot = 0
    
    def callback():
        
        global beta, rot, indices, U, B1, B2, vectors_complex, vectors, posis, F_involved, vectors_complex_unrot
        
        psim.PushItemWidth(150)

        # psim.TextUnformatted("Some sample text")
        # psim.TextUnformatted("An important value: {}".format(42))
        psim.Separator()
        
        indices_new = indices
        changed_idx = False

        for i, index in enumerate(indices):
            changed, index = psim.InputInt(f"Index {i}", index, step=1, step_fast=3) 
            
            if changed:
                indices_new[i] = index
                changed_idx = True
        
        changed_rot, rot = psim.SliderFloat("Rotation", rot, v_min=-np.pi, v_max=np.pi)
        
        changed_beta, beta = psim.InputFloat("beta", beta, step=0.25, step_fast=1)
        Beta = np.full(U.shape[0], beta)
        
        if(psim.Button("Randomise Beta")):
            Beta = np.random.uniform(-1, 1, U.shape[0])
            changed_beta = True
        
        if changed_idx or changed_beta:
            if changed_idx:
                U = mesh.corner_field(singularities, indices_new, v_init, z_init)
                
            coeffs, coeffs_singular, coeffs_subdivided = mesh.reconstruct_linear_from_corners(
                U, Beta = Beta
            )
                
            posis, vectors_complex, F_involved = mesh.sample_field(
                coeffs, coeffs_singular, coeffs_subdivided, num_samples=3, margin=0.15, 
                singular_detail=True, num_samples_detail=12, margin_detail=0.05
            )

            B1, B2 = mesh.B1[F_involved], mesh.B2[F_involved]
            
            vectors_complex_unrot = vectors_complex.copy()
                
        if changed_rot:
            vectors_complex = np.exp(1j * rot) * vectors_complex_unrot
                
        vectors = vectors_complex.real[:, None] * B1 + vectors_complex.imag[:, None] * B2
            
        vectors /= np.linalg.norm(vectors, axis=1)[:, None]
        
        ps_field = ps.register_point_cloud("Field_sample", posis, enabled=True, radius=0)
        ps_field.add_vector_quantity('Field', vectors, enabled=True, length=0.0075)

        psim.PopItemWidth()

    ps.set_user_callback(callback)
    ps.show()
            

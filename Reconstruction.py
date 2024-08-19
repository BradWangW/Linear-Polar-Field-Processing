import numpy as np
import polyscope as ps
from Functions.Auxiliary import *
from Functions.Mesh import Triangle_mesh
# from Functions.NonContractibleBases import *
import os
import polyscope.imgui as psim


# Increase the recursion limit
np.set_printoptions(threshold=np.inf)


if __name__ == '__main__':
    
    V, F = load_off_file(os.path.join('..', 'data', 'spherers.off'))
    E = obtain_E(F)

    mesh = Triangle_mesh(V, F)
    
    mesh.initialise_field_processing()
    
    # singularities = np.array([
    #     0.2 * V[F[70, 0]] + 0.2 * V[F[70, 1]] + 0.6 * V[F[70, 2]],
    #     0.6 * V[F[70, 0]] + 0.2 * V[F[70, 1]] + 0.2 * V[F[70, 2]],
    #     V[F[205, 0]],
    #     0.2 * V[F[205, 0]] + 0.2 * V[F[205, 1]] + 0.6 * V[F[205, 2]],
    #     0.2 * V[F[100, 0]] + 0.2 * V[F[100, 1]] + 0.6 * V[F[100, 2]],
    #     0.6 * V[F[103, 0]] + 0.2 * V[F[103, 1]] + 0.2 * V[F[103, 2]],
    #     (V[E[100, 0]] + V[E[100, 1]])/2,
    #     0.2 * V[F[0, 0]] + 0.2 * V[F[0, 1]] + 0.6 * V[F[0, 2]],
    #     V[F[300, 0]],
    #     0.2 * V[F[5, 0]] + 0.2 * V[F[5, 1]] + 0.6 * V[F[5, 2]]
    # ])
    # indices = [-2, 2, -1, 1, -1, 1, -1, 1, -1, 1]
    singularities = np.array([
        V[70], 
        0.2 * V[F[70, 0]] + 0.2 * V[F[70, 1]] + 0.6 * V[F[70, 2]]
    ])
    indices = [1, -1]
    # singularities = np.array([])
    # indices = []
    
    non_contractible_indices = None
    
    v_init = 100
    z_init = 1
    
    U = mesh.corner_field(singularities, indices, v_init, z_init, non_contractible_indices)
    
    coeffs, coeffs_singular, coeffs_subdivided = mesh.reconstruct_linear_from_corners(U)
    
    posis, vectors_complex, F_involved = mesh.sample_field(
        coeffs, coeffs_singular, coeffs_subdivided, num_samples=3, margin=0.15, 
        singular_detail=True, num_samples_detail=10, margin_detail=0.05
    )
    
    B1, B2 = mesh.B1[F_involved], mesh.B2[F_involved]
    
    vectors = vectors_complex.real[:, None] * B1 + vectors_complex.imag[:, None] * B2
    
    # vectors /= np.linalg.norm(vectors, axis=1)[:, None]

    ps.init()
    ps_mesh = ps.register_surface_mesh("Input Mesh", V, F, color=(0.95, 0.98, 1))

    ps_mesh.add_scalar_quantity('Gaussian Curvature', mesh.G_V, enabled=False, cmap='blues')

    ps_field = ps.register_point_cloud("Field_sample", posis, enabled=True, radius=0)
    ps_field.add_vector_quantity('Field', vectors, enabled=True, color=(0.03, 0.33, 0.77))
    
    for i in range(len(mesh.E_non_contractible_cycles)):
        ps.register_point_cloud(f'non_contractible cycles {i}', mesh.V[np.array(mesh.E_non_contractible_cycles[i])[:, 0]], enabled=True)
    
    # H1_basis = compute_homology_basis(V, E, F)

    # # Print result
    # for i, basis in enumerate(H1_basis):
    #     print(f"Basis {i}: {basis}")
            
    for f in mesh.F_over_pi:
        ps.register_surface_mesh(f"F_over_pi{f}", mesh.V_subdivided[f], mesh.F_subdivided[f], enabled=True)
    
    if len(singularities) > 0:
        ps.register_point_cloud("singularity marker", singularities, enabled=True, radius=0.0015)
    
    vectors_complex_unrot = vectors_complex.copy()
    rot = 0
    unit_length = False
    non_unit_length = False
    
    def callback():
        
        global rot, indices, U, B1, B2, vectors_complex, vectors, posis, F_involved, vectors_complex_unrot, unit_length, non_unit_length
        
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
        
        if changed_idx:
            U = mesh.corner_field(singularities, indices_new, v_init, z_init)
                
            coeffs, coeffs_singular, coeffs_subdivided = mesh.reconstruct_linear_from_corners(
                U
            )
                
            posis, vectors_complex, F_involved = mesh.sample_field(
                coeffs, coeffs_singular, coeffs_subdivided, num_samples=3, margin=0.15, 
                singular_detail=True, num_samples_detail=10, margin_detail=0.05
            )

            B1, B2 = mesh.B1[F_involved], mesh.B2[F_involved]
            
            vectors_complex_unrot = vectors_complex.copy()
                
        if changed_rot:
            vectors_complex = np.exp(1j * rot) * vectors_complex_unrot
                
        vectors = vectors_complex.real[:, None] * B1 + vectors_complex.imag[:, None] * B2
            
        # vectors /= np.linalg.norm(vectors, axis=1)[:, None]
        
        ps_field = ps.register_point_cloud("Field_sample", posis, enabled=True)
        
        if(psim.Button("Unit length")):
            non_unit_length = False
            unit_length = ~unit_length
        if(psim.Button("No unit length")):
            unit_length = False
            non_unit_length = ~non_unit_length
            
        if unit_length:
            ps_field.add_vector_quantity('Field', vectors/np.linalg.norm(vectors, axis=1)[:, None], enabled=True, length=0.005)
        elif non_unit_length:
            ps_field.add_vector_quantity('Field', vectors, enabled=True, length=0.06)
        else:
            ps_field.add_vector_quantity('Field', vectors, enabled=True)

        psim.PopItemWidth()

    ps.set_user_callback(callback)
    ps.show()
            

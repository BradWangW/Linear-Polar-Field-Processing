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
        0.2 * V[F[200, 0]] + 0.2 * V[F[200, 1]] + 0.6 * V[F[200, 2]],
        0.2 * V[F[100, 0]] + 0.2 * V[F[100, 1]] + 0.6 * V[F[100, 2]],
        0.6 * V[F[103, 0]] + 0.2 * V[F[103, 1]] + 0.2 * V[F[103, 2]],
        (V[E[100, 0]] + V[E[100, 1]])/2,
        0.2 * V[F[0, 0]] + 0.2 * V[F[0, 1]] + 0.6 * V[F[0, 2]]
    ])
    indices = [1, -1, 1, 1, 1, -1, -1, 1]
    v_init = 10
    z_init = 1
    
    field = mesh.vector_field(
        singularities, indices, v_init, z_init
    )
    
    posis, vectors_complex, F_involved = mesh.sample_points_and_vectors(
        field, num_samples=2, margin=0.25, singular_detail=False, return_complex=True
    )
    
    B1, B2 = mesh.B1[F_involved], mesh.B2[F_involved]
    
    vectors = vectors_complex.real[:, None] * B1 + vectors_complex.imag[:, None] * B2
    
    vectors /= np.linalg.norm(vectors, axis=1)[:, None]

    ps.init()
    ps_mesh = ps.register_surface_mesh("Input Mesh", V, F)

    ps_field = ps.register_point_cloud("Field_sample", posis, enabled=True, radius=0)
    ps_field.add_vector_quantity('Field', vectors, enabled=True, length=0.01)
    
    # for f in mesh.F_singular:
    #     ps.register_point_cloud(f"Singularities{f}", np.array(mesh.singularities_f[f]), enabled=True, radius=0.002)
        
    # for f in mesh.F_over_pi:
    #     ps.register_surface_mesh(f"F_over_pi{f}", mesh.V_subdivided[f], mesh.F_subdivided[f], enabled=True)
    
    ps.register_point_cloud("singularity marker", singularities, enabled=True, radius=0.002)
    
    def callback():
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
        
        changed_rot, rot = psim.SliderFloat("Rotation", 0, v_min=-np.pi, v_max=np.pi)
        
        changed_beta, beta = psim.InputInt("beta", 1, step=1, step_fast=3)
        
        if changed_rot or changed_idx or changed_beta:
            if changed_idx or changed_beta:
                field_new = mesh.vector_field(
                    singularities, indices_new, v_init, z_init, Beta=np.full(len(F), beta)
                )
                
                _, vectors_complex_new, _ = mesh.sample_points_and_vectors(
                    field_new, num_samples=2, margin=0.25, singular_detail=False, return_complex=True
                )
                
            else:
                vectors_complex_new = vectors_complex.copy()
                
            if changed_rot:
                vectors_complex_new *= np.exp(1j * rot)
                
            vectors = vectors_complex_new.real[:, None] * B1 + vectors_complex_new.imag[:, None] * B2
                
            vectors /= np.linalg.norm(vectors, axis=1)[:, None]
                
            ps_field.remove_all_quantities()
            ps_field.add_vector_quantity('Field', vectors, enabled=True, length=0.01)

        psim.PopItemWidth()

    ps.set_user_callback(callback)
    ps.show()
            

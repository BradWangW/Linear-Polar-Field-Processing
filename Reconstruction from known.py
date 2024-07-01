import numpy as np
import polyscope as ps
from Functions.Auxiliary import *
from Functions.Mesh import Triangle_mesh


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

    mesh = Triangle_mesh(V, F)

    coeffs_truth = np.zeros((len(F), 1, 2), dtype=complex)

    # Truth coefficient for the linear term
    coeffs_truth[:, 0, 0] = 2 + 3j

    zeros_per_face = np.array([
        [1/3, 1/3, -1/3], 
        (2 * (V[0] + V[2]) - V[3])/3, 
        (2 * (V[0] + V[1]) - V[3])/3, 
        (2 * (V[1] + V[2]) - V[3])/3, 
    ])

    Z_zero_per_face = complex_projection(
        mesh.B1, mesh.B2, mesh.normals, zeros_per_face - V[F[:, 0]], diagonal=True
    )

    coeffs_truth[:, 0, 1] = -coeffs_truth[:, 0, 0] * Z_zero_per_face

    # Compute the fields
    field_truth, field = mesh.vector_field_from_truth(coeffs_truth, singularities, indices)

    points_truth, vectors_truth = sample_points_and_vectors(V, F, field_truth, num_samples=25)
    points, vectors = sample_points_and_vectors(V, F, field, num_samples=25)

    # raise ValueError

    # print(vectors_truth, vectors)

    V_truth = V.copy()
    V_truth[:, [0, 2]] += 2
    points_truth[:, [0, 2]] += 2
    singularities_truth = singularities.copy()
    singularities_truth[:, [0, 2]] += 2

    # normalise the vectors
    vectors_truth = vectors_truth / np.linalg.norm(vectors_truth, axis=1)[:, None]
    vectors = vectors / np.linalg.norm(vectors, axis=1)[:, None]

    ps.init()
    ps_mesh_truth = ps.register_surface_mesh("Truth Mesh", V_truth, F)
    ps_mesh = ps.register_surface_mesh("Input Mesh", V, F)

    ps_field_truth = ps.register_point_cloud("Centroids_truth", points_truth, enabled=True)
    ps_field_truth.add_vector_quantity('Samples_truth', vectors_truth, enabled=True)

    ps_field = ps.register_point_cloud("Centroids", points, enabled=True)
    ps_field.add_vector_quantity('Samples', vectors, enabled=True)
    
    ps.register_point_cloud("singularity marker", singularities, enabled=True)
    ps.register_point_cloud("singularity_truth marker", singularities_truth, enabled=True)

    ps.show()
            

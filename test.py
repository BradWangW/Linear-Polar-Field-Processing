import numpy as np
import polyscope as ps

# Initialize Polyscope
ps.init()

# Example mesh: simple triangular mesh
V = np.array([
    [0, 0, 0],  # Vertex 0
    [1, 0, 0],  # Vertex 1
    [0, 1, 0],  # Vertex 2
    [1, 1, 0]   # Vertex 3
])

F = np.array([
    [0, 1, 2],  # Face 0
    [1, 3, 2]   # Face 1
])

# Register the mesh
ps_mesh = ps.register_surface_mesh("Input Mesh", V, F)

# Define a function to compute the vector field at a point in barycentric coordinates
def vector_field(bary_coords, face_index):
    # Example vector field: rotate vectors based on barycentric coordinate positions
    u, v, w = bary_coords
    return np.array([u - v, w - u, v - w])

# Sample points within each face using barycentric coordinates
def sample_points_and_vectors(V, F, num_samples=10):
    points = []
    vectors = []
    for i, face in enumerate(F):
        v0, v1, v2 = V[face]
        for j in range(num_samples):
            for k in range(num_samples - j):
                # Barycentric coordinates
                u = j / num_samples
                v = k / num_samples
                w = 1 - u - v

                # Interpolate to get the 3D point in the face
                point = u * v0 + v * v1 + w * v2
                vector = vector_field([u, v, w], i)

                points.append(point)
                vectors.append(vector)
    
    return np.array(points), np.array(vectors)

# Sample the vector field at multiple points within each face
points, vectors = sample_points_and_vectors(V, F)

# Register the sampled points as a point cloud
ps_point_cloud = ps.register_point_cloud("Sampled Points", points)

# Register the vector field at these points
ps_point_cloud.add_vector_quantity("Sampled Vector Field", vectors, enabled=True)
 
# Show the scene
ps.show()
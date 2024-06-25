import numpy as np
from mayavi import mlab
from scipy.spatial import Delaunay

# Define vertices of the tetrahedron
vertices = np.array([
    [1, 1, 1],
    [-1, -1, 1],
    [-1, 1, -1],
    [1, -1, -1]
])

# Define triangular faces of the tetrahedron
faces = np.array([
    [0, 1, 2],
    [0, 2, 3],
    [0, 3, 1],
    [1, 3, 2]
])

# Function to compute vector field with singularities of index 1
def vector_field(positions):
    singularities = np.array([
        [0.5, 0.5, 0.5],
        [-0.5, -0.5, 0.5]
    ])
    
    vectors = np.zeros_like(positions, dtype=np.float64)  # Ensure vectors are float64
    
    for i, pos in enumerate(positions):
        for singularity in singularities:
            vec = singularity - pos
            vectors[i] += vec / np.linalg.norm(vec)**3
    
    return vectors

# Generate mesh for the tetrahedron
tri = Delaunay(vertices)

# Compute vectors at mesh vertices
vectors = vector_field(vertices)

# Plot the tetrahedron with vector field using mayavi
fig = mlab.figure(bgcolor=(1, 1, 1))

# Plot tetrahedron faces
for face in faces:
    # Define triangles for mlab.triangular_mesh
    triangles = np.array([face])
    triangle = mlab.triangular_mesh(vertices[:, 0], vertices[:, 1], vertices[:, 2], triangles,
                                    scalars=vertices[:, 2], figure=fig)

# Plot vector field
mlab.quiver3d(vertices[:, 0], vertices[:, 1], vertices[:, 2],
              vectors[:, 0], vectors[:, 1], vectors[:, 2],
              line_width=3, scale_factor=0.2, figure=fig)

mlab.show()

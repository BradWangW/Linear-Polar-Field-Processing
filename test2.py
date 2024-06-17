import numpy as np
import pyvista as pv
from pyvista import examples

# Load a built-in example mesh
mesh = examples.download_cow()  # You can replace this with your mesh

# Create a synthetic vector field for demonstration
# This is just a simple vector field for example purposes
vectors = np.zeros((mesh.n_points, 3))
vectors[:, 0] = np.sin(mesh.points[:, 1])
vectors[:, 1] = np.cos(mesh.points[:, 0])
mesh['vectors'] = vectors  # Add vector field to the mesh

# Create a streamlines filter
streamlines, src = mesh.streamlines_from_source(
    source=10.0,
    vectors='vectors',
    integrator_type='rk45',  # Use Runge-Kutta 45 integration method
    max_time=100.0,          # Maximum time for integration
    initial_step_length=0.5, # Initial step length
    terminal_speed=0.01      # Integration stops if speed falls below this value
)

# Plot the mesh and the streamlines
p = pv.Plotter()
p.add_mesh(mesh, color='lightgrey', opacity=0.5)
p.add_mesh(streamlines.tube(radius=0.1), color='blue')  # Use tubes to visualize streamlines
p.show()

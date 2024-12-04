import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Define the complex vector field function
def f(z, r, t):
    a = r
    b = (1-r) * np.exp(1j * t)
    return a * z + b * np.conj(z)

# Generate the grid of points
num_directions = 30  # Number of polar axes directions
num_points = 10 # Number of points per axis

# Generate points in polar coordinates
angles = np.linspace(0, 2 * np.pi, num_directions, endpoint=False)
radii = np.linspace(0.1, 2, num_points)  # Avoiding the origin to prevent singularity

# Convert polar coordinates to Cartesian coordinates
X = []
Y = []
for angle in angles:
    X.extend(radii * np.cos(angle))
    Y.extend(radii * np.sin(angle))

X = np.array(X)
Y = np.array(Y)

# X = np.linspace(-2, 2, 15)
# Y = np.linspace(-2, 2, 15)

# X, Y = np.meshgrid(X, Y)

Z = X + 1j * Y

# Parameters
r = 1  # Example complex number for a
t = 0

# Normalise the field
F = f(Z, r, t)
F = F / np.abs(F)

# Calculate the initial vector field
U = np.real(F)
V = np.imag(F)

# Create the plot
fig, ax = plt.subplots()
q = ax.quiver(X, Y, U, V)
ax.set_title('Complex Vector Field f(z) = az + b conj(z) where b = a r exp(i t)')
ax.set_xlabel('Re(z)')
ax.set_ylabel('Im(z)')

# Adjust the subplots region to leave some space for the sliders
plt.subplots_adjust(left=0.1, bottom=0.25)

# Add sliders for r and t
ax_r = plt.axes([0.1, 0.1, 0.65, 0.03])
ax_t = plt.axes([0.1, 0.05, 0.65, 0.03])

slider_r = Slider(ax_r, 'r', -1, 2, valinit=r)
slider_t = Slider(ax_t, 't', 0.0, 2 * np.pi, valinit=t)

# Update function to be called when sliders are changed
def update(val):
    r = slider_r.val
    t = slider_t.val
    
    F = f(Z, r, t)
    # F = F / np.abs(F)
    
    U = np.real(F)
    V = np.imag(F)
    q.set_UVC(U, V)
    fig.canvas.draw_idle()

# Connect the update function to the sliders
slider_r.on_changed(update)
slider_t.on_changed(update)

plt.show()

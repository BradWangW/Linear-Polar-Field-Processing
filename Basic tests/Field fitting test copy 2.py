import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np

z1 = 1
z2 = 1 * np.exp(1j * 2 * np.pi / 3)
z3 = 1 * np.exp(1j * 4 * np.pi / 3)
u1 = 1 * np.exp(1j)
u2 = 1 * np.exp(1j * np.pi)
u3 = 1 * np.exp(1j * 3 * np.pi / 2)

# Define the complex vector field function
def uv_f(Z, a, I):
    f = (a * Z) ** I
    f = f / np.abs(f)

    return np.real(f), np.imag(f)

def get_params(u, v, w):
    
    a = (u * u1 + v * u2 + w * u3) / (u * z1 + v * z2 + w * z3)

    return a
    
# Generate the grid of points
num_coor = 30  # Number of axes directions
range_coor = 3  # Range of the axes

X = np.linspace(-range_coor, range_coor, num_coor)
Y = np.linspace(-range_coor, range_coor, num_coor)

X, Y = np.meshgrid(X, Y)    
Z = X + 1j * Y

# Parameters
u = 0.33
v = 0.33
w = 1 - u - v
I = 1

a = get_params(u, v, w)

# Calculate the initial vector field
U, V = uv_f(Z, a, I)

X_cor = np.array([z1.real, z2.real, z3.real])
Y_cor = np.array([z1.imag, z2.imag, z3.imag])
U_cor = np.array([u1.real, u2.real, u3.real])
V_cor = np.array([u1.imag, u2.imag, u3.imag])

# Create the plot
fig, ax = plt.subplots()
q = ax.quiver(X, Y, U, V)
q_cor = ax.quiver(X_cor, Y_cor, U_cor, V_cor, color='r')

ax.set_title(f'', color='red')
ax.set_xlabel('Re(z)')
ax.set_ylabel('Im(z)')
ax.set_ylim(-range_coor, range_coor)
ax.set_xlim(-range_coor, range_coor)

# Adjust the subplots region to leave some space for the sliders
plt.subplots_adjust(left=0.1, bottom=0.40)

# Add sliders for r and t
ax_u = plt.axes([0.1, 0.1, 0.65, 0.03])
ax_v = plt.axes([0.1, 0.15, 0.65, 0.03])
ax_I = plt.axes([0.1, 0.2, 0.65, 0.03])

slider_u = Slider(ax_u, 'u', -1, 1, valinit=u)
slider_v = Slider(ax_v, 'v', -1, 1, valinit=v)
slider_I = Slider(ax_I, 'I', -5, 5, valinit=I)

# Update function to be called when sliders are changed
def update(val):
    u = slider_u.val
    v = slider_v.val
    I = slider_I.val
    
    a = get_params(u, v, 1 - u - v)
    U, V = uv_f(Z, a, I)
    q.set_UVC(U, V)

    fig.canvas.draw_idle()

# Connect the update function to the sliders
slider_u.on_changed(update)
slider_v.on_changed(update)
slider_I.on_changed(update)

plt.show()


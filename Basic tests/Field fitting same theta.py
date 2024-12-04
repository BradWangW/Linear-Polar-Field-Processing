import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib import rcParams

# Set global plot style
rcParams.update({
    "text.usetex": False,  # Use LaTeX for text rendering
    "font.family": "serif",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,  # High-resolution plots
    "axes.grid": False,  # Enable grid for better readability
    "grid.alpha": 0.3,  # Make grid lines subtle
    "savefig.dpi": 300,
})

# Parameters
z0 = 1
z1 = -1 + 1j
z2 = -1 - 1j

arg0 = 0
theta_01 = np.pi * 2 / 3
theta_02 = np.pi * 4 / 3
u = 1 / 3
v = 1 / 3
w = 1 - u - v
singularity = z0 * u + z1 * v + z2 * w

# Functions
def scale_func(a, z):
    return np.abs(a * z + (1 - a) * np.conj(z))

def recontructed_zero(z0, z1, z2, arg0, theta_01, theta_02, singularity):
    z0_relative = np.abs(z0 - singularity)
    z1_relative = np.abs(z1 - singularity) * np.exp(1j * (np.angle(z1 - singularity) - np.angle(z0 - singularity)))
    z2_relative = np.abs(z2 - singularity) * np.exp(1j * (np.angle(z2 - singularity) - np.angle(z0 - singularity)))

    scale0 = np.abs(z0_relative)
    A = np.array([[1, np.tan(theta_01)], [1, np.tan(theta_02)]])
    b = np.array([
        z1_relative.real * np.tan(theta_01) / (2 * z1_relative.imag) + 1 / 2,
        z2_relative.real * np.tan(theta_02) / (2 * z2_relative.imag) + 1 / 2,
    ])

    a_re, a_im = np.linalg.solve(A, b)
    a = a_re + a_im * 1j

    scale1 = scale_func(a, z1_relative)
    scale2 = scale_func(a, z2_relative)

    u0 = scale0 * np.exp(1j * arg0)
    u1 = scale1 * np.exp(1j * (arg0 + theta_01))
    u2 = scale2 * np.exp(1j * (arg0 + theta_02))

    A_coeff = np.array([
        [z0.real, -z0.imag, z0.real, z0.imag, 1, 0],
        [z0.imag, z0.real, -z0.imag, z0.real, 0, 1],
        [z1.real, -z1.imag, z1.real, z1.imag, 1, 0],
        [z1.imag, z1.real, -z1.imag, z1.real, 0, 1],
        [z2.real, -z2.imag, z2.real, z2.imag, 1, 0],
        [z2.imag, z2.real, -z2.imag, z2.real, 0, 1],
    ], dtype=float)
    b_coeff = np.array([u0.real, u0.imag, u1.real, u1.imag, u2.real, u2.imag])

    coeff = np.linalg.solve(A_coeff, b_coeff)
    return coeff, u0, u1, u2

def field(Z, coeff):
    a = coeff[0] + 1j * coeff[1]
    b = coeff[2] + 1j * coeff[3]
    c = coeff[4] + 1j * coeff[5]
    f = a * Z + b * np.conj(Z) + c
    print(np.abs(a)/np.abs(b))
    return np.real(f), np.imag(f)

# Grid setup
num_coor = 15
range_coor = 2

X = np.linspace(-range_coor, range_coor, num_coor)
Y = np.linspace(-range_coor, range_coor, num_coor)
X, Y = np.meshgrid(X, Y)
Z = X + 1j * Y

coeff, u0, u1, u2 = recontructed_zero(z0, z1, z2, arg0, theta_01, theta_02, singularity)
U, V = field(Z, coeff)

# Plot
fig, ax = plt.subplots(figsize=(6, 6))
q = ax.quiver(X, Y, U, V, color='darkblue', scale=30, pivot="middle", alpha=0.8)
sin_true = ax.scatter(
    singularity.real, singularity.imag, color='orange', s=50, label=r'$\text{Singularity}$'
)
corners = ax.quiver(
    [z0.real, z1.real, z2.real], [z0.imag, z1.imag, z2.imag],
    [u0.real, u1.real, u2.real], [u0.imag, u1.imag, u2.imag],
    color='red', scale=30, pivot="middle", alpha=0.7
)
triangle = ax.plot(
    [z0.real, z1.real, z2.real, z0.real],
    [z0.imag, z1.imag, z2.imag, z0.imag],
    color='green', linestyle='--', linewidth=1.5, alpha=0.8, label=r'$\text{Triangle}$'
)

ax.set_title(r'\textbf{Vector Field}', pad=10)
ax.set_xlabel(r'$\Re(z)$')
ax.set_ylabel(r'$\Im(z)$')
ax.set_xlim(-range_coor, range_coor)
ax.set_ylim(-range_coor, range_coor)
ax.legend(loc='upper right', frameon=False)

plt.subplots_adjust(left=0.15, bottom=0.35)

# Sliders
ax_u = plt.axes([0.15, 0.1, 0.65, 0.03])
ax_v = plt.axes([0.15, 0.15, 0.65, 0.03])
slider_u = Slider(ax_u, r'$u$', 0, 1, valinit=u)
slider_v = Slider(ax_v, r'$v$', 0, 1, valinit=v)

def update(val):
    u = slider_u.val
    v = slider_v.val
    w = 1 - u - v
    singularity = z0 * u + z1 * v + z2 * w

    coeff, u0, u1, u2 = recontructed_zero(z0, z1, z2, arg0, theta_01, theta_02, singularity)
    U, V = field(Z, coeff)
    q.set_UVC(U, V)
    sin_true.set_offsets([singularity.real, singularity.imag])
    corners.set_UVC([u0.real, u1.real, u2.real], [u0.imag, u1.imag, u2.imag])
    fig.canvas.draw_idle()

slider_u.on_changed(update)
slider_v.on_changed(update)

plt.show()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np

z0 = 1
z1 = -1 + 1j
z2 = -1 - 1j

arg0 = 1
arg1 = 1 + 2 * np.pi / 3
arg2 = 1 + 4 * np.pi / 3
u = 1/3
v = 1/3
w = 1 - u - v
singularity = z0 * u + z1 * v + z2 * w

def scale_func(a, z):
    return np.abs(
        a * z + (1 - a) * np.conj(z)
    )
    
def zero(a, b, c):
    A = np.array([
        [a.real+b.real, -a.imag+b.imag],
        [a.imag+b.imag, a.real-b.real]
    ])
    b = np.array([
        -c.real,
        -c.imag
    ])
    
    return np.linalg.solve(A, b)

# Define the complex vector field function
def recontructed_zero(z0, z1, z2, arg0, arg1, arg2, singularity):
    z0_relative = np.abs(z0 - singularity)
    z1_relative = np.abs(z1 - singularity) * np.exp(1j * (np.angle(z1 - singularity) - np.angle(z0 - singularity)))
    z2_relative = np.abs(z2 - singularity) * np.exp(1j * (np.angle(z2 - singularity) - np.angle(z0 - singularity)))
    
    print(z0_relative, z1_relative, z2_relative)
    
    theta_01 = np.angle(np.exp(1j * (arg1 - arg0)))
    theta_02 = np.angle(np.exp(1j * (arg2 - arg0)))
    
    scale0 = np.abs(z0_relative)
    
    A = np.array([
        [1, np.tan(theta_01)],
        [1, np.tan(theta_02)]
    ])
    b = np.array([
        z1_relative.real * np.tan(theta_01) / (2 * z1_relative.imag) + 1/2, 
        z2_relative.real * np.tan(theta_02) / (2 * z2_relative.imag) + 1/2
    ])
    
    if np.abs(z1_relative.imag) < 1e-6 or np.abs(z2_relative.imag) < 1e-6:
        raise ValueError('The imaginary part of the relative coordinates of the corners is too small')
    
    if np.linalg.cond(A) > 1e8:
        raise ValueError("Matrix A is ill-conditioned")
    
    a_re, a_im = np.linalg.solve(A, b)
    a = a_re + a_im * 1j
    
    scale1 = scale_func(a, z1_relative)
    scale2 = scale_func(a, z2_relative)
    
    u0 = scale0 * np.exp(1j * arg0)
    u1 = scale1 * np.exp(1j * arg1)
    u2 = scale2 * np.exp(1j * arg2)
    
    A_coeff = np.array([
        [z0.real, -z0.imag, z0.real, z0.imag, 1, 0],
        [z0.imag, z0.real, -z0.imag, z0.real, 0, 1],
        [z1.real, -z1.imag, z1.real, z1.imag, 1, 0],
        [z1.imag, z1.real, -z1.imag, z1.real, 0, 1],
        [z2.real, -z2.imag, z2.real, z2.imag, 1, 0],
        [z2.imag, z2.real, -z2.imag, z2.real, 0, 1]
    ], dtype=float)
    b_coeff = np.array([
        u0.real, u0.imag, u1.real, u1.imag, u2.real, u2.imag
    ])
    
    coeff = np.linalg.solve(A_coeff, b_coeff)
    
    a = coeff[0] + 1j * coeff[1]
    b = coeff[2] + 1j * coeff[3]
    c = coeff[4] + 1j * coeff[5]
    
    singularity_reconstructed = (b * np.conjugate(c) - c * np.conjugate(a)) /\
        (np.abs(a)**2 - np.abs(b)**2)
        
    if np.abs(np.abs(a)**2 - np.abs(b)**2) < 1e-6:
        raise ValueError("Denominator near zero in singularity calculation")

    # singularity_reconstructed = (coeff[1] * np.conjugate(coeff[2]) - coeff[2] * np.conjugate(coeff[0])) /\
    #     (np.abs(coeff[0])**2 - np.abs(coeff[1])**2)
    scale_sin = np.abs(coeff[0] * singularity + coeff[1] * np.conjugate(singularity) + coeff[2])
    # scale_cor0 = np.abs(coeff[0] * z0 + coeff[1] * np.conjugate(z0) + coeff[2])
    # scale_cor1 = np.abs(coeff[0] * z1 + coeff[1] * np.conjugate(z1) + coeff[2])
    # scale_cor2 = np.abs(coeff[0] * z2 + coeff[1] * np.conjugate(z2) + coeff[2])
    
    # max_scale_ratio = scale_sin / np.min([scale_cor0, scale_cor1, scale_cor2])
    
    print(scale_sin)
    
    return coeff, singularity_reconstructed, u0, u1, u2

def field(Z, coeff):
    a = coeff[0] + 1j * coeff[1]
    b = coeff[2] + 1j * coeff[3]
    c = coeff[4] + 1j * coeff[5]
    
    f = a * Z + b * np.conj(Z) + c
    # f = f / np.abs(f)
    
    return np.real(f), np.imag(f)
    
# Generate the grid of points
num_coor = 20  # Number of axes directions
range_coor = 2  # Range of the axes

X = np.linspace(-range_coor, range_coor, num_coor)
Y = np.linspace(-range_coor, range_coor, num_coor)

X, Y = np.meshgrid(X, Y)    
Z = X + 1j * Y

coeff, singularity_reconstructed, u0, u1, u2 = recontructed_zero(z0, z1, z2, arg0, arg1, arg2, singularity)

# Calculate the initial vector field
U, V = field(Z, coeff)

# X_cor = np.array([z0.real, z1.real, z2.real])
# Y_cor = np.array([z0.imag, z1.imag, z2.imag])
# U_cor = np.array([u0.real, u1.real, u2.real])
# V_cor = np.array([u0.imag, u1.imag, u2.imag])

# Create the plot
fig, ax = plt.subplots()
q = ax.quiver(X, Y, U, V)
sin_true = ax.scatter(singularity.real, singularity.imag, 
                      color='b', alpha=0.5, 
                      label='True singularity')
sin_point = ax.scatter(singularity_reconstructed.real, 
                       singularity_reconstructed.imag, 
                       color='r', alpha=0.5, 
                       label='Reconstructed singularity')
corners = ax.quiver([z0.real, z1.real, z2.real],
                    [z0.imag, z1.imag, z2.imag],
                    [u0.real, u1.real, u2.real],
                    [u0.imag, u1.imag, u2.imag],
                    color='r', alpha=0.5)
triangle = ax.plot([z0.real, z1.real, z2.real, z0.real],
                     [z0.imag, z1.imag, z2.imag, z0.imag],
                     color='r', alpha=0.5)
# q_cor = ax.quiver(X_cor, Y_cor, U_cor, V_cor, color='r')

ax.set_title(f'', color='red')
ax.set_xlabel('Re(z)')
ax.set_ylabel('Im(z)')
ax.set_ylim(-range_coor, range_coor)
ax.set_xlim(-range_coor, range_coor)

# Adjust the subplots region to leave some space for the sliders
plt.subplots_adjust(left=0.1, bottom=0.40)

# Add sliders for u and v
ax_u = plt.axes([0.1, 0.1, 0.65, 0.03])
ax_v = plt.axes([0.1, 0.15, 0.65, 0.03])

slider_u = Slider(ax_u, 'u', 0, 1, valinit=u)
slider_v = Slider(ax_v, 'v', 0, 1, valinit=v)

# Update function to be called when sliders are changed
def update(val):
    u = slider_u.val
    v = slider_v.val
    w = 1 - u - v
    singularity = z0 * u + z1 * v + z2 * w
    
    coeff, singularity_reconstructed, u0, u1, u2 = recontructed_zero(z0, z1, z2, arg0, arg1, arg2, singularity)
    U, V = field(Z, coeff)
    q.set_UVC(U, V)
    sin_true.set_offsets([singularity.real, singularity.imag])
    sin_point.set_offsets([singularity_reconstructed.real, singularity_reconstructed.imag])
    corners.set_UVC([u0.real, u1.real, u2.real],
                    [u0.imag, u1.imag, u2.imag])

    fig.canvas.draw_idle()

# Connect the update function to the sliders
slider_u.on_changed(update)
slider_v.on_changed(update)

ax.legend()

plt.show()


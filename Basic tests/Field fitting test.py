import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np

# Define the complex vector field function
def uv_f(Z, a, b, c):
    f = a * Z + b * np.conj(Z) + c
    f = f / np.abs(f)

    return np.real(f), np.imag(f)

def get_params(z1, z2_len, z2_angle, theta_01, theta_02, u1_len, u2_len):
    u1 = u1_len * np.exp(1j * theta_01)
    u2 = u2_len * np.exp(1j * theta_02)
    z2 = z2_len * np.exp(1j * z2_angle)
    lhs = np.array([
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
        [z1, 0, z1, 0, 1, 0],
        [0, z1, 0, z1, 0, 1],
        [z2.real, -z2.imag, z2.real, z2.imag, 1, 0],
        [z2.imag, z2.real, -z2.imag, z2.real, 0, 1]
    ])
    rhs = np.array([
        1, 0, u1.real, u1.imag, u2.real, u2.imag
    ])

    soln = np.linalg.solve(lhs, rhs)
    a = soln[0] + 1j * soln[1]
    b = soln[2] + 1j * soln[3]
    c = soln[4] + 1j * soln[5]

    return a, b, c

def corner_XYUV(z1, z2_len, z2_angle, theta_01, theta_02):
    Z_cor = np.array([0, z1, z2_len * np.exp(1j * z2_angle)])
    Vec_cor = np.array([1, np.exp(1j * theta_01), np.exp(1j * theta_02)])
    
    return np.real(Z_cor), np.imag(Z_cor), np.real(Vec_cor), np.imag(Vec_cor)

def corner_perpendicular_lines(X_lines, X_cor, Y_cor, U_cor, V_cor):
    slopes = V_cor / U_cor
    Y_lines = slopes[:, None] * (X_lines[None, :] - X_cor[:, None]) + Y_cor[:, None]
    return Y_lines
    
# Generate the grid of points
num_coor = 30  # Number of axes directions
range_coor = 3  # Range of the axes

X = np.linspace(-range_coor, range_coor, num_coor)
Y = np.linspace(-range_coor, range_coor, num_coor)

X, Y = np.meshgrid(X, Y)    
Z = X + 1j * Y

# Parameters
z1 = 1 
z2_len = 1
z2_angle = np.pi / 4
theta_01 = np.pi / 4
theta_02 = -np.pi / 4
u1_len = 1
u2_len = 1

a, b, c = get_params(z1, z2_len, z2_angle, theta_01, theta_02, u1_len, u2_len)

# Calculate the initial vector field
U, V = uv_f(Z, a, b, c)

X_cor, Y_cor, U_cor, V_cor = corner_XYUV(z1, z2_len, z2_angle, theta_01, theta_02)
X_lines = np.array([-range_coor, range_coor])
Y_lines = corner_perpendicular_lines(X_lines, X_cor, Y_cor, U_cor, V_cor)

# Create the plot
fig, ax = plt.subplots()
q = ax.quiver(X, Y, U, V)
q_cor = ax.quiver(X_cor, Y_cor, U_cor, V_cor, color='r')
line0 = ax.plot(X_lines, Y_lines[0], color='r')
line1 = ax.plot(X_lines, Y_lines[1], color='r')
line2 = ax.plot(X_lines, Y_lines[2], color='r')
triangle = ax.plot([X_cor[0], X_cor[1], X_cor[2], X_cor[0]],[Y_cor[0], Y_cor[1], Y_cor[2], Y_cor[0]])

U_f_cor, V_f_cor = uv_f(X_cor + 1j * Y_cor, a, b, c)
orientation_accordance = np.nan_to_num(np.arccos(
    np.real(np.conjugate(U_cor + 1j * V_cor) * (U_f_cor + 1j * V_f_cor)) /
    (np.abs(U_cor + 1j * V_cor) * np.abs(U_f_cor + 1j * V_f_cor))
)) < np.pi / 2
if np.sum(orientation_accordance) == 3 or np.sum(orientation_accordance) == 0:
    ax.set_title(f'Orientation accordance: {np.sum(orientation_accordance)}, \
                    Corner 1: {orientation_accordance[1]}, \
                    Corner 2: {orientation_accordance[2]}',
                 color='green')
else:
    ax.set_title(f'Orientation accordance: {np.sum(orientation_accordance)}, \
                    Corner 1: {orientation_accordance[1]}, \
                    Corner 2: {orientation_accordance[2]}',
                 color='red')
    
zero = (b - np.conj(a))/(np.abs(a)**2 - np.abs(b)**2)
zero_pt = ax.scatter(zero.real, zero.imag, color='black')

circle_centre = - U_cor[1] / V_cor[1] * (z1/2 - X_cor[1]) + Y_cor[1]
radius_to_zero = ax.plot([z1/2, zero.real], [circle_centre, zero.imag], color='black')

ax.set_title(f'', color='red')
ax.set_xlabel('Re(z)')
ax.set_ylabel('Im(z)')
ax.set_ylim(-range_coor, range_coor)
ax.set_xlim(-range_coor, range_coor)

# Adjust the subplots region to leave some space for the sliders
plt.subplots_adjust(left=0.1, bottom=0.40)

# Add sliders for r and t
ax_z1 = plt.axes([0.1, 0.1, 0.65, 0.03])
ax_z2_len = plt.axes([0.1, 0.15, 0.65, 0.03])
ax_z2_angle = plt.axes([0.1, 0.2, 0.65, 0.03])
ax_theta_01 = plt.axes([0.1, 0.25, 0.65, 0.03])
ax_theta_02 = plt.axes([0.1, 0.3, 0.65, 0.03])
ax_u1_len = plt.axes([0.1, 0.35, 0.65, 0.03])
ax_u2_len = plt.axes([0.1, 0.4, 0.65, 0.03])

slider_z1 = Slider(ax_z1, 'z1', 0.1, 3, valinit=z1)
slider_z2_len = Slider(ax_z2_len, 'z2_len', 0.1, 3, valinit=z2_len)
slider_z2_angle = Slider(ax_z2_angle, 'z2_angle', -np.pi, np.pi, valinit=z2_angle)
slider_theta_01 = Slider(ax_theta_01, 'theta_01', -np.pi, np.pi, valinit=theta_01)
slider_theta_02 = Slider(ax_theta_02, 'theta_02', -np.pi, np.pi, valinit=theta_02)
slider_u1_len = Slider(ax_u1_len, 'u1_len', 0.1, 3, valinit=u1_len)
slider_u2_len = Slider(ax_u2_len, 'u2_len', 0.1, 3, valinit=u2_len)

# Update function to be called when sliders are changed
def update(val):
    z1 = slider_z1.val
    z2_len = slider_z2_len.val
    z2_angle = slider_z2_angle.val
    theta_01 = slider_theta_01.val
    theta_02 = slider_theta_02.val
    u1_len = slider_u1_len.val
    u2_len = slider_u2_len.val
    
    a, b, c = get_params(z1, z2_len, z2_angle, theta_01, theta_02, u1_len, u2_len)
    U, V = uv_f(Z, a, b, c)
    q.set_UVC(U, V)

    X_cor, Y_cor, U_cor, V_cor = corner_XYUV(z1, z2_len, z2_angle, theta_01, theta_02)
    Y_lines = corner_perpendicular_lines(X_lines, X_cor, Y_cor, U_cor, V_cor)
    q_cor.set_offsets(np.c_[X_cor.ravel(), Y_cor.ravel()])
    q_cor.set_UVC(U_cor, V_cor)
    line0[0].set_data(X_lines, Y_lines[0])
    line1[0].set_data(X_lines, Y_lines[1])
    line2[0].set_data(X_lines, Y_lines[2])
    triangle[0].set_data([X_cor[0], X_cor[1], X_cor[2], X_cor[0]],[Y_cor[0], Y_cor[1], Y_cor[2], Y_cor[0]])

    U_f_cor, V_f_cor = uv_f(X_cor + 1j * Y_cor, a, b, c)
    orientation_accordance = np.nan_to_num(np.arccos(
        np.real(np.conjugate(U_cor + 1j * V_cor) * (U_f_cor + 1j * V_f_cor)) /
        (np.abs(U_cor + 1j * V_cor) * np.abs(U_f_cor + 1j * V_f_cor))
    )) < np.pi / 2
    if np.sum(orientation_accordance) == 3 or np.sum(orientation_accordance) == 0:
        ax.set_title(f'Orientation accordance: {np.sum(orientation_accordance)}, \
                    Corner 1: {orientation_accordance[1]}, \
                    Corner 2: {orientation_accordance[2]}',
                    color='green')
    else:
        ax.set_title(f'Orientation accordance: {np.sum(orientation_accordance)}, \
                    Corner 1: {orientation_accordance[1]}, \
                    Corner 2: {orientation_accordance[2]}',
                    color='red')
    zero = (b - np.conj(a))/(np.abs(a)**2 - np.abs(b)**2)
    zero_pt.set_offsets(np.c_[zero.real, zero.imag])
    
    circle_centre = - U_cor[1] / V_cor[1] * (z1/2 - X_cor[1]) + Y_cor[1]
    radius_to_zero[0].set_data([z1/2, zero.real], [circle_centre, zero.imag])

    fig.canvas.draw_idle()

# Connect the update function to the sliders
slider_z1.on_changed(update)
slider_z2_len.on_changed(update)
slider_z2_angle.on_changed(update)
slider_theta_01.on_changed(update)
slider_theta_02.on_changed(update)
slider_u1_len.on_changed(update)
slider_u2_len.on_changed(update)

plt.show()


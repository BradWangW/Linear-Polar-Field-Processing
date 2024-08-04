import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np

# Define the complex vector field function
def uv_f(Z, a_re, a_im, b_re=1, b_im=0):
    f = (a_re + 1j * a_im) * Z + (b_re + 1j * b_im)
    f = f / np.abs(f)

    return np.real(f), np.imag(f)

def get_params(z1, z2_len, z2_angle, theta_01, theta_02, theta_0):
    lhs = np.array([
        [z1 * np.sin(theta_01), -z1 * np.cos(theta_01)], 
        [z2_len * np.sin(theta_02 - z2_angle), -z2_len * np.sin(theta_02 + z2_angle)]
    ])
    rhs = np.array([-np.sin(theta_01), -np.sin(theta_02)])

    soln = np.linalg.solve(lhs, rhs)
    a = soln[0] + 1j * soln[1]
    b = 1
    a *= np.exp(1j * theta_0); b *= np.exp(1j * theta_0)

    return a.real, a.imag, b.real, b.imag

def corner_XYUV(z1, z2_len, z2_angle, theta_01, theta_02, theta_0):
    Z_cor = np.array([0, z1, z2_len * np.exp(1j * z2_angle)])
    Vec_cor = np.array([1, np.exp(1j * theta_01), np.exp(1j * theta_02)])
    Vec_cor *= np.exp(1j * theta_0)
    
    return np.real(Z_cor), np.imag(Z_cor), np.real(Vec_cor), np.imag(Vec_cor)

def corner_perpendicular_lines(X_lines, X_cor, Y_cor, U_cor, V_cor):
    slopes = - U_cor / V_cor
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
theta_0 = 1

a_re, a_im, b_re, b_im = get_params(z1, z2_len, z2_angle, theta_01, theta_02, theta_0)

# Calculate the initial vector field
U, V = uv_f(Z, a_re, a_im, b_re, b_im)

X_cor, Y_cor, U_cor, V_cor = corner_XYUV(z1, z2_len, z2_angle, theta_01, theta_02, theta_0)
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

U_f_cor, _ = uv_f(X_cor + 1j * Y_cor, a_re, a_im, b_re)
orientation_accordance = np.sign(U_f_cor) == np.sign(U_cor)
if np.sum(orientation_accordance) == 3 or np.sum(orientation_accordance) == 0:
    ax.set_title(f'Orientation accordance: {np.sum(orientation_accordance)}, \
                 Corner 1: {np.sign(U_f_cor[1]) == np.sign(U_cor[1])}, \
                 Corner 2: {np.sign(U_f_cor[2]) == np.sign(U_cor[2])}',
                 color='green')
else:
    ax.set_title(f'Orientation accordance: {np.sum(orientation_accordance)}, \
                 Corner 1: {np.sign(U_f_cor[1]) == np.sign(U_cor[1])}, \
                 Corner 2: {np.sign(U_f_cor[2]) == np.sign(U_cor[2])}',
                 colorcolor='red')
    
zero = -(b_re + 1j * b_im) / (a_re + 1j * a_im)
print(zero.real, zero.imag)
zero_pt = ax.scatter(zero.real, zero.imag, color='black')

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
ax_theta_0 = plt.axes([0.1, 0.35, 0.65, 0.03])

slider_z1 = Slider(ax_z1, 'z1', 0.1, 3, valinit=z1)
slider_z2_len = Slider(ax_z2_len, 'z2_len', 0.1, 3, valinit=z2_len)
slider_z2_angle = Slider(ax_z2_angle, 'z2_angle', -np.pi, np.pi, valinit=z2_angle)
slider_theta_01 = Slider(ax_theta_01, 'theta_01', -np.pi, np.pi, valinit=theta_01)
slider_theta_02 = Slider(ax_theta_02, 'theta_02', -np.pi, np.pi, valinit=theta_02)
slider_theta_0 = Slider(ax_theta_0, 'theta_0', -np.pi, np.pi, valinit=theta_0)

# Update function to be called when sliders are changed
def update(val):
    z1 = slider_z1.val
    z2_len = slider_z2_len.val
    z2_angle = slider_z2_angle.val
    theta_01 = slider_theta_01.val
    theta_02 = slider_theta_02.val
    theta_0 = slider_theta_0.val
    
    a_re, a_im, b_re, b_im = get_params(z1, z2_len, z2_angle, theta_01, theta_02, theta_0)
    U, V = uv_f(Z, a_re, a_im, b_re, b_im)
    q.set_UVC(U, V)

    X_cor, Y_cor, U_cor, V_cor = corner_XYUV(z1, z2_len, z2_angle, theta_01, theta_02, theta_0)
    Y_lines = corner_perpendicular_lines(X_lines, X_cor, Y_cor, U_cor, V_cor)
    q_cor.set_offsets(np.c_[X_cor.ravel(), Y_cor.ravel()])
    q_cor.set_UVC(U_cor, V_cor)
    line0[0].set_data(X_lines, Y_lines[0])
    line1[0].set_data(X_lines, Y_lines[1])
    line2[0].set_data(X_lines, Y_lines[2])
    triangle[0].set_data([X_cor[0], X_cor[1], X_cor[2], X_cor[0]],[Y_cor[0], Y_cor[1], Y_cor[2], Y_cor[0]])

    U_f_cor, _ = uv_f(X_cor + 1j * Y_cor, a_re, a_im, b_re, b_im)
    orientation_accordance = np.sign(U_f_cor) == np.sign(U_cor)
    if np.sum(orientation_accordance) == 3 or np.sum(orientation_accordance) == 0:
        ax.set_title(f'Orientation accordance: {np.sum(orientation_accordance)}, \
                    Corner 1: {np.sign(U_f_cor[1]) == np.sign(U_cor[1])}, \
                    Corner 2: {np.sign(U_f_cor[2]) == np.sign(U_cor[2])}',
                    color='green')
    else:
        ax.set_title(f'Orientation accordance: {np.sum(orientation_accordance)}, \
                    Corner 1: {np.sign(U_f_cor[1]) == np.sign(U_cor[1])}, \
                    Corner 2: {np.sign(U_f_cor[2]) == np.sign(U_cor[2])}',
                    color='red')
        
    zero = -(b_re + 1j * b_im) / (a_re + 1j * a_im)
    zero_pt.set_offsets(np.c_[zero.real, zero.imag])

    fig.canvas.draw_idle()

# Connect the update function to the sliders
slider_z1.on_changed(update)
slider_z2_len.on_changed(update)
slider_z2_angle.on_changed(update)
slider_theta_01.on_changed(update)
slider_theta_02.on_changed(update)
slider_theta_0.on_changed(update)

plt.show()


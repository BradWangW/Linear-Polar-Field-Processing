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
theta_0 = 0.001

def params(z1, z2_len, z2_angle, theta_01, theta_02, theta_0):

    theta_12 = theta_02 - theta_01
    z3_len = np.sqrt(z1**2 + z2_len**2 - 2 * z1 * z2_len * np.cos(z2_angle))
    
    z1s = [z1, z3_len, z2_len]
    z2_lens = [z2_len, z1, z3_len]
    z2_angles = [
        z2_angle,
        np.arccos((z1**2 + z3_len**2 - z2_len**2) / (2 * z1 * z3_len)),
        np.arccos((z3_len**2 + z2_len**2 - z1**2) / (2 * z3_len * z2_len))
    ]
    theta_01s = [theta_01, theta_12, -theta_02]
    theta_02s = [theta_02, -theta_01, -theta_12]
    theta_0s = [theta_0, theta_0, theta_0]
    
    print(np.sum(z2_angles))
    
    return z1s, z2_lens, z2_angles, theta_01s, theta_02s, theta_0s

def compute_quantities(z1s, z2_lens, z2_angles, theta_01s, theta_02s, theta_0s):
    Us = []
    Vs = []
    X_cors = []
    Y_cors = []
    U_cors = []
    V_cors = []
    X_liness = []
    Y_liness = []
    orientation_accordances = []
    zeros = []
    circle_centres = []
    
    for i in range(3):
        z1, z2_len, z2_angle, theta_01, theta_02, theta_0 = z1s[i], z2_lens[i], z2_angles[i], theta_01s[i], theta_02s[i], theta_0s[i]
        
        a_re, a_im, b_re, b_im = get_params(z1, z2_len, z2_angle, theta_01, theta_02, theta_0)

        # Calculate the initial vector field
        U, V = uv_f(Z, a_re, a_im, b_re, b_im)

        X_cor, Y_cor, U_cor, V_cor = corner_XYUV(z1, z2_len, z2_angle, theta_01, theta_02, theta_0)
        X_lines = np.array([-range_coor, range_coor])
        Y_lines = corner_perpendicular_lines(X_lines, X_cor, Y_cor, U_cor, V_cor)

        U_f_cor, V_f_cor = uv_f(X_cor + 1j * Y_cor, a_re, a_im, b_re)
        orientation_accordance = np.nan_to_num(np.arccos(
            np.real(np.conjugate(U_cor + 1j * V_cor) * (U_f_cor + 1j * V_f_cor)) /
            (np.abs(U_cor + 1j * V_cor) * np.abs(U_f_cor + 1j * V_f_cor))
        )) < np.pi / 2
        
        zero = -(b_re + 1j * b_im) / (a_re + 1j * a_im)
        
        circle_centre = - U_cor[1] / V_cor[1] * (z1/2 - X_cor[1]) + Y_cor[1]
        
        Us.append(U)
        Vs.append(V)
        X_cors.append(X_cor)
        Y_cors.append(Y_cor)
        U_cors.append(U_cor)
        V_cors.append(V_cor)
        X_liness.append(X_lines)
        Y_liness.append(Y_lines)
        orientation_accordances.append(orientation_accordance)
        zeros.append(zero)
        circle_centres.append(circle_centre)
    
    return Us, Vs, X_cors, Y_cors, U_cors, V_cors, X_liness, Y_liness, orientation_accordances, zeros, circle_centres
        

z1s, z2_lens, z2_angles, theta_01s, theta_02s, theta_0s = params(z1, z2_len, z2_angle, theta_01, theta_02, theta_0)

Us, Vs, X_cors, Y_cors, U_cors, V_cors, X_liness, Y_liness, orientation_accordances, zeros, circle_centres = compute_quantities(z1s, z2_lens, z2_angles, theta_01s, theta_02s, theta_0s)

# Create the plot
fig, axes = plt.subplots(1, 3)

qs = []
q_cors = []
line0s = []
line1s = []
line2s = []
line2_paras = []
triangles = []
zero_pts = []
radius_to_zeros = []

for i in range(3):
    ax = axes[i]
    U, V, X_cor, Y_cor, U_cor, V_cor, X_lines, Y_lines, orientation_accordance, zero, circle_centre = Us[i], Vs[i], X_cors[i], Y_cors[i], U_cors[i], V_cors[i], X_liness[i], Y_liness[i], orientation_accordances[i], zeros[i], circle_centres[i]
    
    qs.append(ax.quiver(X, Y, U, V))
    q_cors.append(ax.quiver(X_cor, Y_cor, U_cor, V_cor, color='r'))
    line0s.append(ax.plot(X_lines, Y_lines[0], color='r'))
    line1s.append(ax.plot(X_lines, Y_lines[1], color='r'))
    line2s.append(ax.plot(X_lines, Y_lines[2], color='r'))
    line2_paras.append(ax.plot(X_lines, (U_cor/V_cor)[2] * (X_lines - X_cor[2]) + Y_cor[2], color='k'))
    triangles.append(ax.plot([X_cor[0], X_cor[1], X_cor[2], X_cor[0]],[Y_cor[0], Y_cor[1], Y_cor[2], Y_cor[0]]))
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
        
    zero_pts.append(ax.scatter(zero.real, zero.imag, color='black'))

    radius_to_zeros.append(ax.plot([z1/2, zero.real], [circle_centre, zero.imag], color='black'))

    ax.set_title(f'', color='red')
    ax.set_xlabel('Re(z)')
    ax.set_ylabel('Im(z)')
    ax.set_ylim(-range_coor, range_coor)
    ax.set_xlim(-range_coor, range_coor)

# Adjust the subplots region to leave some space for the sliders
plt.subplots_adjust(left=0.1, bottom=0.45)

# Add sliders for r and t
ax_z1 = plt.axes([0.1, 0.1, 0.65, 0.03])
ax_z2_len = plt.axes([0.1, 0.15, 0.65, 0.03])
ax_z2_angle = plt.axes([0.1, 0.2, 0.65, 0.03])
ax_theta_01 = plt.axes([0.1, 0.25, 0.65, 0.03])
ax_theta_02 = plt.axes([0.1, 0.3, 0.65, 0.03])
ax_theta_0 = plt.axes([0.1, 0.35, 0.65, 0.03])

slider_z1 = Slider(ax_z1, 'z1', 0.1, 3, valinit=z1)
slider_z2_len = Slider(ax_z2_len, 'z2_len', 0.1, 3, valinit=z2_len)
slider_z2_angle = Slider(ax_z2_angle, 'z2_angle', -np.pi/2, np.pi/2, valinit=z2_angle)
slider_theta_01 = Slider(ax_theta_01, 'theta_01', -np.pi/2, np.pi/2, valinit=theta_01)
slider_theta_02 = Slider(ax_theta_02, 'theta_02', -np.pi/2, np.pi/2, valinit=theta_02)
slider_theta_0 = Slider(ax_theta_0, 'theta_0', -np.pi, np.pi, valinit=theta_0)

# Update function to be called when sliders are changed
def update(val):
    z1 = slider_z1.val
    z2_len = slider_z2_len.val
    z2_angle = slider_z2_angle.val
    theta_01 = slider_theta_01.val
    theta_02 = slider_theta_02.val
    theta_0 = slider_theta_0.val
    
    z1s, z2_lens, z2_angles, theta_01s, theta_02s, theta_0s = params(z1, z2_len, z2_angle, theta_01, theta_02, theta_0)
    
    Us, Vs, X_cors, Y_cors, U_cors, V_cors, X_liness, Y_liness, orientation_accordances, zeros, circle_centres = compute_quantities(z1s, z2_lens, z2_angles, theta_01s, theta_02s, theta_0s)
    
    for i in range(3):
        qs[i].set_UVC(Us[i], Vs[i])
        q_cors[i].set_offsets(np.c_[X_cors[i], Y_cors[i]])
        q_cors[i].set_UVC(U_cors[i], V_cors[i])
        line0s[i][0].set_data(X_liness[i], Y_liness[i][0])
        line1s[i][0].set_data(X_liness[i], Y_liness[i][1])
        line2s[i][0].set_data(X_liness[i], Y_liness[i][2])
        line2_paras[i][0].set_data(X_liness[i], (V_cors[i]/U_cors[i])[2] * (X_liness[i] - X_cors[i][2]) + Y_cors[i][2])
        triangles[i][0].set_data([X_cors[i][0], X_cors[i][1], X_cors[i][2], X_cors[i][0]],[Y_cors[i][0], Y_cors[i][1], Y_cors[i][2], Y_cors[i][0]])
        
        if np.sum(orientation_accordances[i]) == 3 or np.sum(orientation_accordances[i]) == 0:
            axes[i].set_title(
                f'Orientation accordance: {np.sum(orientation_accordances[i])}, Corner 1: {orientation_accordances[i][1]}, Corner 2: {orientation_accordances[i][2]}',
                color='green')
        else:
            axes[i].set_title(
                f'Orientation accordance: {np.sum(orientation_accordances[i])}, Corner 1: {orientation_accordances[i][1]}, Corner 2: {orientation_accordances[i][2]}',
                color='red')
        
        zero_pts[i].set_offsets(np.c_[zeros[i].real, zeros[i].imag])
        
        radius_to_zeros[i][0].set_data([z1s[i]/2, zeros[i].real], [circle_centres[i], zeros[i].imag])

    fig.canvas.draw_idle()

# Connect the update function to the sliders
slider_z1.on_changed(update)
slider_z2_len.on_changed(update)
slider_z2_angle.on_changed(update)
slider_theta_01.on_changed(update)
slider_theta_02.on_changed(update)
slider_theta_0.on_changed(update)

plt.show()


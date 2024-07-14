import numpy as np
import matplotlib.pyplot as plt

# Define the complex vector field function
def complex_vector_field(z):
    return (z - 1) * (z + 1)**(-1)

# Create a grid of points in the complex plane
x = np.linspace(-3, 3, 30)
y = np.linspace(-3, 3, 30)
X, Y = np.meshgrid(x, y)
Z = X + 1j * Y

# Parameters for the complex vector field
a = 1 + 1j
b = 0.5 - 0.5j
c = 0.1 + 0.1j

# Compute the vector field at each point in the grid
F = complex_vector_field(Z)
F = F/np.abs(F)

# Separate the vector field into real and imaginary parts for plotting
U = np.real(F)
V = np.imag(F)

# Plot the vector field using quiver
plt.figure(figsize=(8, 8))
plt.quiver(X, Y, U, V, color='blue')
plt.xlabel('Re(z)')
plt.ylabel('Im(z)')
plt.title('Complex Vector Field $f(z) = az + b\overline{z} + c$')
plt.grid()
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.show()

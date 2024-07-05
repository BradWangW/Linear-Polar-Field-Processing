import numpy as np
import matplotlib.pyplot as plt

# Define the complex function f(z) = (z + 1 + i) / (z - 1 - i)
def f(z):
    # return (z + 1 + 1j) *  1/((z - 1 - 1j))
    return (z+1)**(-1)


N = 30

# Create a grid of points in the complex plane
x = np.linspace(-3, 3, N)
y = np.linspace(-3, 3, N)
X, Y = np.meshgrid(x, y)
Z = X + 1j * Y

# Calculate the function values at each point on the grid
F = f(Z)

# Normalise as we are only interested in the direction of the vectors
F = F / np.abs(F)

# Extract the real and imaginary parts
U = np.real(F)
V = np.imag(F)

# Plot the vector field using quiver
plt.figure(figsize=(8, 8))
plt.quiver(X, Y, U, V, color='b')

# Add labels and title
plt.xlabel('Real part')
plt.ylabel('Imaginary part')
plt.title('Vector Field for f(z) = (z + 1 + i) / (z - 1 - i)')
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.grid(True)
plt.gca().set_aspect('equal', adjustable='box')

# Show the plot
plt.show()

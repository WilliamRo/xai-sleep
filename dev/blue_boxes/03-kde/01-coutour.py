import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Example data
np.random.seed(42)
x1 = np.random.normal(0, 1, 1000)  # X data
x2 = np.random.normal(0, 1, 1000)  # Y data

# Estimate the kernel density
kernel = gaussian_kde(np.vstack([x1, x2]))

# Evaluate the density on a grid
xmin, xmax = x1.min() - 1, x1.max() + 1  # Define X range
ymin, ymax = x2.min() - 1, x2.max() + 1  # Define Y range
X, Y = np.meshgrid(np.linspace(xmin, xmax, 100), np.linspace(ymin, ymax, 100))
positions = np.vstack([X.ravel(), Y.ravel()])  # Stack grid positions to evaluate the kernel
Z = np.reshape(kernel(positions).T, X.shape)  # Evaluate the kernel density

# Plot
fig, ax = plt.subplots(figsize=(8, 6))
contour = ax.contour(X, Y, Z, levels=10, cmap='viridis')  # Generate contour lines
ax.set_title('Kernel Density Estimation Contour')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
plt.colorbar(contour, ax=ax)  # Optional: Add a colorbar to show density levels
plt.show()
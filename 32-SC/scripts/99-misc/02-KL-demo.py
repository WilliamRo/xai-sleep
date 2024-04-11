import numpy as np
import scipy
import sklearn.metrics



# Generate two clusters of data points following a normal distribution
P = np.random.normal(0, 1, 10000)
Q = np.random.normal(0, 2, 10000)

# Calculate distance between P and Q
d = scipy.stats.wasserstein_distance(P, Q)

print(d)



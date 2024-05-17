from collections import OrderedDict
from hypnomics.hypnoprints import extract_hypnoprints_from_hypnocloud
from roma import io
from scipy import stats


import numpy as np
import os



# -----------------------------------------------------------------------------
# Configs
# -----------------------------------------------------------------------------
# Select .sg files
data_dir = r'../../features/'
time_reso = [30, 15, 10, 6, 2][0]
file_name = f'C2-dt{time_reso}.clouds'

# 0: placebo, 1: Temazepam
targets = np.load(r'../../features/targets.npz')['y']


clouds = io.load_file(os.path.join(data_dir, file_name), verbose=True)
cd1, cd2 = OrderedDict(), OrderedDict()
for i, cloud in enumerate(clouds):
  if i // 2 + 1 in (3, 4, 5, 14, 15, 17, 18, 19, 21, 22): continue
  cloud_key = f'ST-{i // 2 + 1:02d}-'
  if targets[i] == 0: cloud_key += 'P'
  else: cloud_key += 'T'

  cd = cd1 if targets[i] == 0 else cd2
  cd[cloud_key] = cloud

# Extract hypnoprints
channels = ['EEG Fpz-Cz', 'EEG Pz-Oz']

# ----------------------------------------------------------------------------
# Extract features
# ----------------------------------------------------------------------------
X1 = np.vstack([extract_hypnoprints_from_hypnocloud(cloud)
                for cloud in cd1.values()])
X2 = np.vstack([extract_hypnoprints_from_hypnocloud(cloud)
                for cloud in cd2.values()])
X = np.concatenate([X1, X2], axis=0)

# Normalize (important)
mu, sigma = X.mean(axis=0), X.std(axis=0)
X1 = (X1 - mu) / sigma
X2 = (X2 - mu) / sigma

# ----------------------------------------------------------------------------
# Define dist_mat
# ----------------------------------------------------------------------------
N, D = X1.shape
X1, X2 = np.expand_dims(X1, 1), np.expand_dims(X2, 0)
X1 = np.broadcast_to(X1, [N, N, D])
X2 = np.broadcast_to(X2, [N, N, D])
M = np.linalg.norm(X1 - X2, axis=2)

acc = np.mean(np.argmin(M, axis=1) == np.arange(N))

# Do t-test
# Get diagonal elements
inner = M[np.diag_indices(N)]
# Get non-diagonal elements
inter = M[~np.eye(N, dtype=bool)]
t_stat, p_val = stats.ttest_ind(
  inner, inter[:N], equal_var=False, alternative='less')
# ----------------------------------------------------------------------------
# Visualization
# ----------------------------------------------------------------------------
from pictor import Pictor
p = Pictor.image_viewer('Matrices')
# p.plotters[0].set('vmin', 0)
# p.plotters[0].set('vmax', 1)
p.plotters[0].set('cmap', 'plasma')
p.plotters[0].set('color_bar', True)
p.plotters[0].set('title', True)
p.objects = [M]
p.labels = [f'{N} subjects, Accuracy = {acc:.3f}, P = {p_val:.4f}']
p.show()



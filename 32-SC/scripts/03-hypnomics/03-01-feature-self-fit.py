import numpy as np

from roma import console
from sc.sc_agent import SCAgent
from sc.tf_optimizer import MatOptimizer



# ----------------------------------------------------------------------------
# Load valid fp_dict
# ----------------------------------------------------------------------------
sca = SCAgent(data_key='dual;alpha')
sca.report_data_info()


# ----------------------------------------------------------------------------
# Fit
# ----------------------------------------------------------------------------
def get_distance_matrix(sca: SCAgent, channel, dim1_tuple, dim2_tuple, N=999):
  # Fetch data
  fea_dicts_1, fea_dicts_2 = sca.get_feature_dicts(
    channel, dim1_tuple, dim2_tuple)
  uni_subs = sca.beta_uni_subs[:N]

  # Optimization
  D = fea_dicts_1[uni_subs[0]].shape[0]
  w = np.ones((1, D))
  X1 = np.vstack([fea_dicts_1[s] for s in uni_subs])
  X2 = np.vstack([fea_dicts_2[s] for s in uni_subs])

  X1, X2 = w * X1, w * X2
  X1 = X1 / np.linalg.norm(X1, axis=1, keepdims=True)
  X2 = X2 / np.linalg.norm(X2, axis=1, keepdims=True)

  dist_mat = X1 @ X2.T

  return dist_mat


# ----------------------------------------------------------------------------
# Analyze features
# ----------------------------------------------------------------------------
channels = ['EEG Fpz-Cz', 'EEG Pz-Oz']
max_freqs = [20, 30]

# Calculate distance matrices
N = 20
for i, pid in enumerate(sca.beta_uni_subs[:N]):
  console.supplement(f'{i+1:02d}/{N:02d} {pid}')

matrices, labels = [], []
for chn in channels:
  dim2_tuple = ('BM02-AMP', 'pool_size', 128)
  for mf in max_freqs:
    dim1_tuple = ('BM01-FREQ', 'max_freq', mf)

    matrix = get_distance_matrix(sca, chn, dim1_tuple, dim2_tuple, N)
    matrices.append(matrix)
    labels.append(f'{chn} (F{mf})')

# Visualize distance matrices
from pictor import Pictor
p = Pictor.image_viewer('Hypno Analysis')
p.plotters[0].set('vmin', 0)
p.plotters[0].set('vmax', 1)
p.plotters[0].set('cmap', 'plasma')
p.plotters[0].set('color_bar', True)
p.plotters[0].set('title', True)
p.objects = matrices
p.labels = labels
p.show()



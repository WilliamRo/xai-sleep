from roma import console
from sc.sc_agent import SCAgent
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import numpy as np



# ----------------------------------------------------------------------------
# Load valid fp_dict
# ----------------------------------------------------------------------------
sca = SCAgent(data_key='dual;alpha')
sca.report_data_info()

# ----------------------------------------------------------------------------
# Configure
# ----------------------------------------------------------------------------
channel = ['EEG Fpz-Cz', 'EEG Pz-Oz']
max_freq = [20, 30][0]

dim1_tuple = ('BM01-FREQ', 'max_freq', max_freq)
dim2_tuple = ('BM02-AMP', 'pool_size', 128)

N = 67
# ----------------------------------------------------------------------------
# Fit
# ----------------------------------------------------------------------------
data, targets, pids = sca.get_feature_targets(
  channel, dim1_tuple, dim2_tuple, version='v2')
data, targets = data[:2*N], targets[:2*N]
data = np.vstack(data)

pca = PCA().fit_transform(data)
tsne = TSNE(n_components=2, random_state=0, init='pca').fit_transform(data)

# ----------------------------------------------------------------------------
# Visualization
# ----------------------------------------------------------------------------
plt.figure(figsize=(12, 6))
for sp, tt, data in zip((121, 122), ('t-SNE', 'PCA'), (tsne, pca)):
  d_min, d_max = np.min(data, 0), np.max(data, 0)
  data = (data - d_min) / (d_max - d_min)
  ax: plt.Axes = plt.subplot(sp)
  for i in range(data.shape[0]):
    color = plt.cm.Set1(targets[i] / N)
    if i % 2 == 0:
      ax.plot(data[i:i+2, 0], data[i:i+2, 1], '-', color=color)
    ax.text(data[i, 0], data[i, 1], s=pids[i],
            color=color, fontdict={'weight': 'bold', 'size': 9})
  ax.set_title(tt)

plt.show()

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
X1, X2 = (X1 - mu) / sigma, (X2 - mu) / sigma

#
x_keys = list(extract_hypnoprints_from_hypnocloud(
  list(cd1.values())[0], return_dict=True).keys())
objects = []
for xk, x1, x2 in zip(x_keys, X1.T, X2.T):
  objects.append({'P': x1, 'T': x2, 'label': xk})

# ----------------------------------------------------------------------------
# Hypothesis test
# ----------------------------------------------------------------------------
from pictor import Pictor
import matplotlib.pyplot as plt

p = Pictor(title='Hypothesis test')
p.objects = [objects]

def multi_boxes(ax: plt.Axes, x):
  results = []
  for pkg in x:
    xk, P, T = pkg['label'], pkg['P'], pkg['T']
    delta = T - P
    alternative = 'greater' if delta.mean() > 0 else 'less'
    _, p_val = stats.ttest_1samp(delta, 0, alternative=alternative)
    results.append((xk, p_val, delta))

  results = sorted(results, key=lambda r: r[1])

  results = results[::-1]
  ax.plot([0, 0], [-1, len(results) + 1], 'r-')
  ax.boxplot([r[2] for r in results], showfliers=False, vert=False)
  ax.set_yticklabels([f'{r[0]}, p={r[1]:.4f}' for r in results])

p.add_plotter(multi_boxes)
p.add_plotter(multi_boxes)
p.show()

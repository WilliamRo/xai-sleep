from collections import OrderedDict
from hypnomics.hypnoprints import extract_hypnoprints_from_hypnocloud
from roma import io


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
p.objects = objects

def plot_box_inter(ax: plt.Axes, x):
  P, T = x['P'], x['T']
  ax.boxplot([P, T], showfliers=False)
  ax.set_xticklabels(['Placebo', 'Temazepam'])

  from scipy import stats
  alternative = 'greater' if P.mean() > T.mean() else 'less'
  t_stat, p_val = stats.ttest_ind(
    P, T, equal_var=False, alternative=alternative)

  ax.set_title(f'{x["label"]}, p = {p_val: .4f}')

def plot_box_inner(ax: plt.Axes, x):
  P, T = x['P'], x['T']
  delta = T - P
  ax.boxplot([delta], showfliers=False)
  ax.set_xticklabels(['Temazepam - Placebo'])

  from scipy import stats
  alternative = 'greater' if delta.mean() > 0 else 'less'
  t_stat, p_val = stats.ttest_1samp(delta, 0, alternative=alternative)
  ax.set_title(f'{x["label"]}, p = {p_val: .4f}')

p.add_plotter(plot_box_inter)
p.add_plotter(plot_box_inner)
p.show()

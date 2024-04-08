import numpy as np

from roma import console
from sc.sc_agent import SCAgent
from scipy import stats



# ----------------------------------------------------------------------------
# Load valid fp_dict
# ----------------------------------------------------------------------------
sca = SCAgent()
sca.report_data_info()
# sca.show_violin_plot()
# sca.visualize_fp_v1()

fp_dict = sca.get_fp_v1_dict()
meta = fp_dict['meta']
subjects, channels, features = meta
M = 20
uni_subs = sorted(list(set([s[:-2] for s in subjects])))[:M]
N = len(uni_subs)

# ----------------------------------------------------------------------------
# Define vector extraction functions
# ----------------------------------------------------------------------------
MIN_SIZE = 5
stages = ['W', 'N1', 'N2', 'N3', 'R']

def estimate_kde(m1, m2):
  values = np.vstack([m1, m2])
  kernel = stats.gaussian_kde(values)
  return kernel

def extract_kde_vector(pid, channel, max_freq):
  kde_dict = {}
  xs = fp_dict[(pid, channel, ('BM01-FREQ', 'max_freq', max_freq))]
  ys = fp_dict[(pid, channel, ('BM02-AMP', 'pool_size', 128))]
  for stage in stages:
    if stage not in xs: continue
    x, y = xs[stage], ys[stage]
    if len(x) < MIN_SIZE: continue
    knl = estimate_kde(x, y)
    kde_dict[stage] = knl

  return kde_dict

# region: Distance Library

def calc_dist_01(d1: dict, d2: dict):
  from scipy.stats import wasserstein_distance_nd

  stages = ['W', 'N1', 'N2', 'N3', 'R']
  v = []
  for stage in stages:
    k1, k2 = d1.get(stage, None), d2.get(stage, None)
    if k1 is None and k2 is None:
      # Case 1: None exists
      d = 0
    elif k1 is None or k2 is None:
      d = 1
    else:
      # Case 2: Both k1 and k2 are not None
      assert isinstance(k1, stats.gaussian_kde)
      assert isinstance(k2, stats.gaussian_kde)
      d = wasserstein_distance_nd(k1.dataset.T, k2.dataset.T)

    # Add to distance vector
    v.append(d)

  labels = ['wasser-W', 'wasser-N1', 'wasser-N2', 'wasser-N3', 'wasser-R']
  return np.array(v), labels

def calc_dist_02(d1: dict, d2: dict):
  stages = ['W', 'N1', 'N2', 'N3', 'R']
  v = []
  for stage in stages:
    k1, k2 = d1.get(stage, None), d2.get(stage, None)
    if k1 is None and k2 is None:
      # Case 1: None exists
      d = [0, 0]
    elif k1 is None or k2 is None:
      d = [2, 2]
    else:
      # Case 2: Both k1 and k2 are not None
      assert isinstance(k1, stats.gaussian_kde)
      assert isinstance(k2, stats.gaussian_kde)
      d = [
        np.linalg.norm([k1.dataset.mean(axis=-1) - k2.dataset.mean(axis=-1)]),
        np.linalg.norm(k1.covariance - k2.covariance)]

    # Add to distance vector
    v.append(d)

  labels = ['dmu-W', 'dsigma-W', 'dmu-N1', 'dsigma-N1', 'dmu-N2', 'dsigma-N2',
            'dmu-N3', 'dsigma-N3', 'dmu-R', 'dsigma-R']
  return np.concatenate(v), labels

# endregion: Distance Library

# exit(777)
# ----------------------------------------------------------------------------
# Analyze features
# ----------------------------------------------------------------------------
channel = ['EEG Fpz-Cz', 'EEG Pz-Oz'][0]
max_freq = [20, 30][0]

# (1) Generate KDE
kde_dicts_1, kde_dicts_2 = {}, {}
console.show_status('Estimating kde vectors ...')
for i, s in enumerate(subjects):
  # console.print_progress(i, len(subjects))
  kde_dict = extract_kde_vector(s, channel, max_freq)
  # Put results into corresponding dict
  pid = s[:-2]
  # assert pid in uni_subs
  tgt_dict = kde_dicts_2 if pid in kde_dicts_1 else kde_dicts_1
  tgt_dict[pid] = kde_dict

assert len(kde_dicts_1) == len(kde_dicts_2)
console.show_status(f'Successfully generated {len(subjects)} kde vectors')

# (2) Calculate distance
method = [
  calc_dist_01,
  calc_dist_02,
][1]
console.show_status('Calculating distance ...')
matrix = None
for i, s1 in enumerate(uni_subs):
  for j, s2 in enumerate(uni_subs):
    console.print_progress(i * N + j, N * N)
    d, labels = method(kde_dicts_1[s1], kde_dicts_2[s2])
    if matrix is None: matrix = np.zeros((len(d), N, N))
    matrix[:, i, j] = d

console.show_status(f'Distance matrix generated')

# (3) Visualize distance matrix
from pictor import Pictor
p = Pictor.image_viewer('Hypno Analysis')
p.plotters[0].set('cmap', 'RdYlGn')
p.plotters[0].set('color_bar', True)
p.plotters[0].set('title', True)
p.objects = matrix
p.labels = labels
p.show()

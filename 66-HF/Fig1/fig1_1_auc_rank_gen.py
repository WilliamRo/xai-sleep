"""A figure of 2D.
X-axis, AUC for subject matching performance.
Y-axis, ranking of (wo upsilon) -> (w upsilon), sorted by difference

Data for plotting is in 66-HF/20
See also 66-HF/02/x21
"""
from hf.sc_tools import get_dual_nebula
from hf.probe_tools import get_probe_keys
from hypnomics.freud.nebula import Nebula
from pictor.xomics.evaluation.roc import ROC
from pictor.xomics.omix import Omix
from roma import io
from typing import OrderedDict
from x_dual_view import PAIRED_LABELS

import os
import numpy as np



# -----------------------------------------------------------------------------
# (1) Configuration
# -----------------------------------------------------------------------------
# (1.1) Target nebula file path
SRC_DIR = r'../data/sleepedfx_sc'
WORK_DIR = r'./data'
# [ 2(x), 5(x), 10(x), 30, ]
TIME_RESOLUTION = 30
NEB_FN = f'SC-{TIME_RESOLUTION}s-KDE-39-probes.nebula'
neb_file_path = os.path.join(SRC_DIR, NEB_FN)

# (1.2) Set channels
CHANNELS = [
  'EEG Fpz-Cz',
  'EEG Pz-Oz'
]
# (1.3) Set probe keys
PROBE_CONFIG = 'ABD'
PROBE_KEYS = get_probe_keys(PROBE_CONFIG)

# -----------------------------------------------------------------------------
# (2) Load paired nebula
# -----------------------------------------------------------------------------
assert os.path.exists(neb_file_path)
nebula: Nebula = Nebula.load(neb_file_path)

nebula.set_labels(PAIRED_LABELS)
neb_1, neb_2 = get_dual_nebula(nebula)
N = len(neb_1.labels)

# -----------------------------------------------------------------------------
# (3) Construct omix
# -----------------------------------------------------------------------------
od = OrderedDict()

CHNL_PROB_PRODUCTS = [(ck, pk) for ck in CHANNELS for pk in PROBE_KEYS]

targets, feature_labels, sample_labels = [], [], []
IJs = [(i, j) for i in range(N) for j in range(N)]
for i, j in IJs:
  targets.append(0 if i == j else 1)
  sample_labels.append(f'({neb_1.labels[i]}, {neb_2.labels[j]})')
target_labels = ['Not Same', 'Same']

features = np.zeros((N * N, len(CHNL_PROB_PRODUCTS)), dtype=np.float32)
# row and col corresponds to data X
for col, (ck, pk) in enumerate(CHNL_PROB_PRODUCTS):
  for conditional in (0, 1):
    MAT_SUFFIX = ['KDE-DISTS-NC', 'KDE-DISTS'][conditional]
    mat_key = f'{ck}-{pk}-{MAT_SUFFIX}'

    kde_dist_dict = nebula.get_from_pocket(mat_key, key_should_exist=True)
    for row, (i, j) in enumerate(IJs):
      label_1, label_2 = neb_1.labels[i], neb_2.labels[j]
      k = (label_1, label_2)
      features[row, col] = kde_dist_dict[k]

    # Calculate AUC and put into od
    auc = ROC(features[:, col], targets).auc
    od[(ck, pk, conditional)] = auc
    print(f'{ck}, {pk}, {conditional}: {auc}')

# -----------------------------------------------------------------------------
# (4) Save data
# -----------------------------------------------------------------------------
SAVE_PATH = os.path.join(WORK_DIR, 'c_nc_auc.od')
io.save_file(od, SAVE_PATH, verbose=True)

feature_labels = None
omix = Omix(features, targets, feature_labels, sample_labels, target_labels,
            data_name=MAT_SUFFIX)
omix.show_in_explorer()

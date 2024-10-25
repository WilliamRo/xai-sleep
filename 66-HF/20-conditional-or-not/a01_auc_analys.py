'''See 02/10'''
from hf.sc_tools import get_paired_sg_labels, get_dual_nebula, get_joint_key
from hypnomics.freud.nebula import Nebula
from hypnomics.hypnoprints.stat_models.model_1 import HypnoModel1
from roma import console
from pictor.xomics.omix import Omix
from x_dual_view import PAIRED_LABELS

import os, time
import numpy as np




# -----------------------------------------------------------------------------
# (1) Configuration
# -----------------------------------------------------------------------------
# (1.1) Target nebula file path
WORK_DIR = r'../data/sleepedfx_sc'
# [ 2(x), 5(x), 10(x), 30, ]
TIME_RESOLUTION = 30
NEB_FN = f'SC-{TIME_RESOLUTION}s-KDE-39-probes.nebula'
neb_file_path = os.path.join(WORK_DIR, NEB_FN)

# (1.2) Set channels
CHANNELS = [
  'EEG Fpz-Cz',
  # 'EEG Pz-Oz'
]

# (1.3) Set probe keys
PROBE_KEYS = [
  'FREQ-20',
  'AMP-1',
  'GFREQ-35',

  'P-TOTAL',
  'RP-DELTA',
  'RP-THETA',
  'RP-ALPHA',
  'RP-BETA',

  'MAG',
  'KURT',
  'ENTROPY',
]

for b1, b2 in [('DELTA', 'TOTAL'), ('THETA', 'TOTAL'), ('ALPHA', 'TOTAL'),
               ('DELTA', 'THETA'), ('DELTA', 'ALPHA'), ('THETA', 'ALPHA')]:
  for stat_key in ['95', 'MIN', 'AVG', 'STD']:
    PROBE_KEYS.append(f'RPS-{b1}_{b2}_{stat_key}')

for b in ['DELTA', 'THETA', 'ALPHA', 'SIGMA']: PROBE_KEYS.append(f'BKURT-{b}')

CHNL_PROB_KEYS = [(ck, pk) for ck in CHANNELS for pk in PROBE_KEYS]

MAT_SUFFIX = ['KDE-DISTS', 'KDE-DISTS-NC'][0]
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
CHNL_PROB_PRODUCTS = [(ck, pk) for ck in CHANNELS for pk in PROBE_KEYS]
features = np.zeros((N * N, len(CHNL_PROB_PRODUCTS)), dtype=np.float32)

targets, feature_labels, sample_labels = [], [], []
IJs = [(i, j) for i in range(N) for j in range(N)]
for i, j in IJs:
  targets.append(1 if i == j else 0)
  sample_labels.append(f'({neb_1.labels[i]}, {neb_2.labels[j]})')
target_labels = ['Not Same', 'Same']

for col, (ck, pk) in enumerate(CHNL_PROB_PRODUCTS):
  mat_key = f'{ck}-{pk}-{MAT_SUFFIX}'

  feature_labels.append(mat_key)

  kde_dist_dict = nebula.get_from_pocket(mat_key, key_should_exist=True)
  for row, (i, j) in enumerate(IJs):
    label_1, label_2 = neb_1.labels[i], neb_2.labels[j]
    k = (label_1, label_2)
    features[row, col] = kde_dist_dict[k]

omix = Omix(features, targets, feature_labels, sample_labels, target_labels,
            data_name=MAT_SUFFIX)
omix.show_in_explorer()


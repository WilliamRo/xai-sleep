from hypnomics.freud.nebula import Nebula
from hypnomics.hypnoprints.extractor import Extractor
from hf.match_lab import MatchLab
from hf.model_helper import gen_dist_mat
from hf.sc_tools import get_dual_nebula
from roma import console
from roma import finder
from x_dual_view import PAIRED_LABELS

import os
import numpy as np



# -----------------------------------------------------------------------------
# (1) Configuration
# -----------------------------------------------------------------------------
WORK_DIR = r'../data/sleepedfx_sc'
CHANNELS = [
  'EEG Fpz-Cz',
  'EEG Pz-Oz'
]

PROBE_KEYS = [
  'FREQ-20', 'AMP-1',
  # 'GFREQ-35',
  'P-TOTAL', 'RP-DELTA', 'RP-THETA', 'RP-ALPHA', 'RP-BETA',
]

# SG_LABELS = ['SC4001E', 'SC4002E']
SG_LABELS = finder.walk(WORK_DIR, type_filter='dir', return_basename=True)[:999]

# [ 2, 5, 10, 30, ]
TIME_RESOLUTION = 30

# NEB_FN = f'SC-{TIME_RESOLUTION}-KDE.nebula'
NEB_FN = f'SC-{TIME_RESOLUTION}-KDE-0730.nebula'
neb_file_path = os.path.join(WORK_DIR, NEB_FN)
# -----------------------------------------------------------------------------
# (2) Load paired nebula
# -----------------------------------------------------------------------------
if os.path.exists(neb_file_path):
  nebula: Nebula = Nebula.load(neb_file_path)
else:
  raise FileNotFoundError(f'Nebula file not found: {neb_file_path}')

nebula.set_labels(PAIRED_LABELS)
neb_1, neb_2 = get_dual_nebula(nebula)
# -----------------------------------------------------------------------------
# (3) Visualize matrix
# -----------------------------------------------------------------------------
N = len(neb_1.labels)
matrices = []
labels = []
n_ck, n_pk = len(CHANNELS), len(PROBE_KEYS)
for ck_i, ck in enumerate(CHANNELS):
  for pk_i, pk in enumerate(PROBE_KEYS):

    mat_key = f'{ck}-{pk}-KDE-DISTS'
    kde_dist_dict = nebula.get_from_pocket(mat_key, key_should_exist=True)

    mat = gen_dist_mat(neb_1, neb_2, kde_dist_dict, mat_key)

    matrices.append(mat)
    labels.append(mat_key)

console.show_status('Matrix constructed.')

# TODO:
extractor = Extractor()
F1 = extractor.extract(neb_1, return_dict=True)
F2 = extractor.extract(neb_2, return_dict=True)

matlab = MatchLab(F1, F2, normalize=1, N=999,
                  neb_1=neb_1, neb_2=neb_2, nebula=nebula)
matlab.select_feature(min_ICC=0.5, verbose=1, set_C=1)

matlab.analyze(matrices=matrices, labels=labels, toolbar=1, omix=True)

from hypnomics.freud.freud import Freud
from hypnomics.freud.nebula import Nebula
from hypnomics.hypnoprints.stat_models.model_1 import HypnoModel1
from hf.sc_tools import get_dual_nebula
from roma import console
from roma import finder

from x_dual_view import PAIRED_LABELS

import os


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

NEB_FN = f'SC-{TIME_RESOLUTION}-KDE-0730.nebula'
neb_file_path = os.path.join(WORK_DIR, NEB_FN)
# -----------------------------------------------------------------------------
# (2) Load paired nebula
# -----------------------------------------------------------------------------
if os.path.exists(neb_file_path):
  nebula: Nebula = Nebula.load(neb_file_path)
else:
  freud = Freud(WORK_DIR)
  nebula = freud.load_nebula(sg_labels=SG_LABELS,
                             channels=CHANNELS,
                             time_resolution=TIME_RESOLUTION,
                             probe_keys=PROBE_KEYS)

nebula.set_labels(PAIRED_LABELS)
neb_1, neb_2 = get_dual_nebula(nebula)
# -----------------------------------------------------------------------------
# (3) Calculate KDE distance matrix of CHANNEL-PK
# -----------------------------------------------------------------------------
# Sanity check
N = len(neb_1.labels)
assert N == len(neb_2.labels)

for PK, CHANNEL in [(pk, ck) for ck in CHANNELS for pk in PROBE_KEYS]:
  console.show_status(f'Estimating `{PK}`-`{CHANNEL}` KDE distance matrix ...')

  # Get kde_dist_dict
  mat_key = f'{CHANNEL}-{PK}-KDE-DISTS'
  kde_dist_dict = nebula.get_from_pocket(
    mat_key, initializer=lambda: {}, local=True)

  hm = HypnoModel1()
  for i, label_1 in enumerate(neb_1.labels):
    for j, label_2 in enumerate(neb_2.labels):
      console.print_progress(index=i * N + j, total=N * N)

      k = (label_1, label_2)
      if k in kde_dist_dict: continue

      key_1, key_2 = [(lb, CHANNEL, PK) for lb in (label_1, label_2)]
      data_1, data_2 = neb_1.data_dict[key_1], neb_2.data_dict[key_2]

      try:
        d = hm.calc_distance(data_1, data_2, key_1, key_2)
        kde_dist_dict[k] = d
      except:
        console.warning(
          f'Failed to calculate KDE distance between {label_1} and {label_2} !')

  # Save nebula
  nebula.save(neb_file_path)

console.show_status('KDE distance matrix calculation done !')
# -----------------------------------------------------------------------------
# (4) Save nebula
# -----------------------------------------------------------------------------
# nebula.save(neb_file_path)

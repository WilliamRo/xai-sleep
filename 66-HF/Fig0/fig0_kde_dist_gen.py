from hf.probe_tools import get_probe_keys
from hf.sc_tools import get_dual_nebula
from hypnomics.freud.nebula import Nebula
from hypnomics.hypnoprints.stat_models.model_1 import HypnoModel1
from roma import console, finder, io
from x_dual_view import PAIRED_LABELS

import os
import time



# -----------------------------------------------------------------------------
# (1) Configuration
# -----------------------------------------------------------------------------
# (1.1) Working directory & patient inclusion
WORK_DIR = r'../data/sleepedfx_sc'
SG_LABELS = finder.walk(WORK_DIR, type_filter='dir', return_basename=True)

# (1.2) TODO: Configure this part
CONDITIONAL = 1
PROBE_CONFIG = 'ABC'
OVERWRITE = 0

CHANNELS = [
  'EEG Fpz-Cz',
  'EEG Pz-Oz'
]

# (1.3) File names and MISC
NEB_FN = f'SC-30s-ABC38.nebula'
NEB_PATH = os.path.join(WORK_DIR, NEB_FN)

PROBE_KEYS = get_probe_keys(PROBE_CONFIG)
CHNL_PROB_KEYS = [(ck, pk) for ck in CHANNELS for pk in PROBE_KEYS]
# -----------------------------------------------------------------------------
# (2) Generate KDE distance
# -----------------------------------------------------------------------------
# (2.1) Load nebula
assert os.path.exists(NEB_PATH)
nebula: Nebula = Nebula.load(NEB_PATH)

# (2.2) Get paired neb
nebula.set_labels(PAIRED_LABELS)
neb_1, neb_2 = get_dual_nebula(nebula)

# -----------------------------------------------------------------------------
# (3) Calculate joint KDE distance matrix of each CHANNEL-PK combination
# -----------------------------------------------------------------------------
# (3.0) Sanity check
N = len(neb_1.labels)
assert N == len(neb_2.labels)

# (3.1) Calculate KDE distance matrix
hm = HypnoModel1()
for ck, pk in CHNL_PROB_KEYS:
  # (3.1.1) Get KDE distance dictionary for (ck, pk, CONDITION)
  mat_key = (ck, pk, CONDITIONAL)
  kde_dist_dict = nebula.get_from_pocket(
    mat_key, initializer=lambda: {}, local=True)

  if len(kde_dist_dict) == N * N:
    console.show_status(f'Joint KDE distance matrix `{mat_key}` already estimated')
    if not OVERWRITE: continue

  # (3.1.2) Fill in KDE distance dictionary for (ck, pk, CONDITION)
  console.show_status(f'Estimating {mat_key} joint KDE distance matrix ...')

  tic = time.time()
  total = N * N - len(kde_dist_dict)
  for i, label_1 in enumerate(neb_1.labels):
    for j, label_2 in enumerate(neb_2.labels):
      # (3.1.2.1) Print progress
      index = i * N + j
      console.print_progress(index=index, total=total)

      # (3.1.2.2) Check distance key
      k = (label_1, label_2)
      if k in kde_dist_dict and not OVERWRITE: continue

      # (3.1.2.3) Fetch data dictionaries
      key_1, key_2 = (label_1, ck, pk), (label_2, ck, pk)
      data_1, data_2 = neb_1.data_dict[key_1], neb_2.data_dict[key_2]

      # (3.1.2.4) Calculate KDE distance for (key_1, key_2)
      try:
        d = hm.calc_distance(data_1, data_2, key_1, key_2,
                             conditional=CONDITIONAL)
        kde_dist_dict[k] = d
      except: console.warning(
        f'Failed to calculate KDE distance between {key_1} and {key_2} !')

      # (3.1.2.-1) Save nebula at three equinoxes
      if any([index == int(total * p) for p in [0.33, 0.66]]):
        nebula.save(NEB_PATH)

  # (3.1.3) Save nebula
  nebula.save(NEB_PATH)
  console.show_status(f'Time elapsed: {time.time() - tic:.3f}s')

console.show_status('KDE distance matrix calculation done !')

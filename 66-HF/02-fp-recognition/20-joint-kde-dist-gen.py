from hf.sc_tools import get_paired_sg_labels, get_dual_nebula, get_joint_key
from hypnomics.freud.nebula import Nebula
from hypnomics.hypnoprints.stat_models.model_1 import HypnoModel1
from roma import console
from x_dual_view import PAIRED_LABELS

import os, time



# -----------------------------------------------------------------------------
# (1) Configuration
# -----------------------------------------------------------------------------
# (1.1) Target nebula file path
WORK_DIR = r'../data/sleepedfx_sc'
# [ 2(x), 5(x), 10(x), 30, ]
TIME_RESOLUTION = 30
NEB_FN = f'SC-{TIME_RESOLUTION}-KDE-0730.nebula'
neb_file_path = os.path.join(WORK_DIR, NEB_FN)

# (1.2) Set estimating configs
CHANNELS = [
  # 'EEG Fpz-Cz',
  'EEG Pz-Oz'
]

PROBE_KEYS = [
  'FREQ-20',
  'AMP-1',
  # 'GFREQ-35',
  # 'P-TOTAL',
  'RP-DELTA', 'RP-THETA', 'RP-ALPHA',
  # 'RP-BETA',
]

CHNL_PROB_KEYS = [(ck, pk) for ck in CHANNELS for pk in PROBE_KEYS]
# -----------------------------------------------------------------------------
# (2) Load paired nebula
# -----------------------------------------------------------------------------
assert os.path.exists(neb_file_path)
nebula: Nebula = Nebula.load(neb_file_path)

nebula.set_labels(PAIRED_LABELS)
neb_1, neb_2 = get_dual_nebula(nebula)

# -----------------------------------------------------------------------------
# (3) Calculate joint KDE distance matrix of each CHANNEL-PK combination
# -----------------------------------------------------------------------------
# Sanity check
N = len(neb_1.labels)
assert N == len(neb_2.labels)

CHNL_PROB_PRODUCTS = [
  (ck1, pk1, ck2, pk2)
  for (ck1, pk1) in CHNL_PROB_KEYS for (ck2, pk2) in CHNL_PROB_KEYS
  if ck1 != ck2 or pk1 != pk2
]

hm = HypnoModel1()
for ck1, pk1, ck2, pk2 in CHNL_PROB_PRODUCTS:
  # Get joint_kde_dist_dict
  reverse_key = get_joint_key(ck2, pk2, ck1, pk1)
  if nebula.in_pocket(reverse_key): continue

  mat_key = get_joint_key(ck1, pk1, ck2, pk2)
  kde_dist_dict = nebula.get_from_pocket(
    mat_key, initializer=lambda: {}, local=True)

  if len(kde_dist_dict) == N * N:
    console.show_status(f'Joint KDE distance matrix `{mat_key}` already estimated')
    continue

  console.show_status(f'Estimating {mat_key} joint KDE distance matrix ...')

  tic = time.time()
  for i, label_1 in enumerate(neb_1.labels):
    for j, label_2 in enumerate(neb_2.labels):
      index, total = i * N + j, N * N
      console.print_progress(index=index, total=total)

      # Save nebula
      if any([index == int(total * p) for p in [0.33, 0.66]]):
        nebula.save(neb_file_path)

      k = (label_1, label_2)
      if k in kde_dist_dict: continue

      key_1_1, key_1_2 = (label_1, ck1, pk1), (label_1, ck2, pk2)
      data_i_1, data_i_2 = neb_1.data_dict[key_1_1], neb_1.data_dict[key_1_2]

      key_2_1, key_2_2 = (label_2, ck1, pk1), (label_2, ck2, pk2)
      data_j_1, data_j_2 = neb_2.data_dict[key_2_1], neb_2.data_dict[key_2_2]

      try:
        d = hm.calc_joint_distance((data_i_1, data_i_2), (data_j_1, data_j_2),
                                   (key_1_1, key_1_2), (key_2_1, key_2_2))
        kde_dist_dict[k] = d
      except:
        console.warning(
          f'Failed to calculate KDE distance between {label_1} and {label_2} !')

  # Save nebula
  nebula.save(neb_file_path)
  console.show_status(f'Time elapsed: {time.time() - tic:.3f}s')

console.show_status('KDE distance matrix calculation done !')


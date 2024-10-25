'''See 02/10'''
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
NEB_FN = f'SC-{TIME_RESOLUTION}s-KDE-39-probes.nebula'
neb_file_path = os.path.join(WORK_DIR, NEB_FN)

# (1.2) Set channels
CHANNELS = [
  'EEG Fpz-Cz',
  'EEG Pz-Oz'
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

CHNL_PROB_PRODUCTS = [(ck, pk) for ck in CHANNELS for pk in PROBE_KEYS]

hm = HypnoModel1()
for ck, pk in CHNL_PROB_PRODUCTS:
  mat_key = f'{ck}-{pk}-KDE-DISTS'
  console.show_status(f'Estimating {mat_key} joint KDE distance matrix ...')

  # Get kde_dist_dict
  kde_dist_dict = nebula.get_from_pocket(
    mat_key, initializer=lambda: {}, local=True)

  hm = HypnoModel1()
  for i, label_1 in enumerate(neb_1.labels):
    for j, label_2 in enumerate(neb_2.labels):
      console.print_progress(index=i * N + j, total=N * N)

      k = (label_1, label_2)

      # TODO: comment this line if you want to re-calculate all
      if k in kde_dist_dict: continue

      key_1, key_2 = [(lb, ck, pk) for lb in (label_1, label_2)]
      data_1, data_2 = neb_1.data_dict[key_1], neb_2.data_dict[key_2]

      try:
        d = hm.calc_distance(data_1, data_2, key_1, key_2)
        kde_dist_dict[k] = d
      except:
        console.warning(
          f'Failed to calculate KDE distance between {label_1} and {label_2} !')

  # Save nebula
  # nebula.save(neb_file_path)

console.show_status('KDE distance matrix calculation done !')
# -----------------------------------------------------------------------------
# (4) Save nebula
# -----------------------------------------------------------------------------
# nebula.save(neb_file_path)

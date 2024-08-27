from hypnomics.hypnoprints.extractor import Extractor
from hypnomics.freud.freud import Freud
from roma import finder
from pictor.xomics.omix import Omix

import numpy as np



# -----------------------------------------------------------------------------
# (1) Configuration
# -----------------------------------------------------------------------------
# Specify the directory containing clouds files
WORK_DIR = r'../data/sleepedfx_sc'

# [ 2, 5, 10, 30, ]
TIME_RESOLUTION = 30

# Specify channels to visualize
CHANNELS = [
  'EEG Fpz-Cz',
  'EEG Pz-Oz'
]

PROBE_KEYS = [
  'FREQ-20',   # 0
  'AMP-1',     # 1
  # 'P-TOTAL',   # 2
  # 'RP-DELTA',  # 3
  # 'RP-THETA',  # 4
  # 'RP-ALPHA',  # 5
  # 'RP-BETA',   # 6
]
PK1 = PROBE_KEYS[0]
PK2 = PROBE_KEYS[1]

# SG_LABELS = ['SC4001E', 'SC4002E']
N = 6
SG_LABELS = finder.walk(WORK_DIR, type_filter='dir', return_basename=True)[:N]
# -----------------------------------------------------------------------------
# (2) Load nebula
# -----------------------------------------------------------------------------
freud = Freud(WORK_DIR)
nebula = freud.load_nebula(sg_labels=SG_LABELS,
                           channels=CHANNELS,
                           time_resolution=TIME_RESOLUTION,
                           probe_keys=PROBE_KEYS)

fake_ages = np.random.randint(20, 100, size=len(nebula.labels))
fake_genders = np.random.choice(['M', 'F'], size=len(nebula.labels))

for i, pid in enumerate(nebula.labels):
  nebula.meta[pid] = {'age': fake_ages[i], 'gender': fake_genders[i]}

if 0:
  from hypnomics.freud.telescopes.telescope import Telescope

  configs = {
    'show_kde': 0,
    'show_scatter': 1,
    'show_vector': 0,
  }
  viewer_configs = {'plotters': 'HA', 'meta_keys': ('age', 'gender')}
  nebula.dual_view(x_key=PK1, y_key=PK2, viewer_configs=viewer_configs,
                   viewer_class=Telescope, **configs)
  exit()

# -----------------------------------------------------------------------------
# (3) Omix construction
# -----------------------------------------------------------------------------
extractor = Extractor()
feature_dict = extractor.extract(nebula, return_dict=True)
features = np.stack([np.array(list(v.values()))
                     for v in feature_dict.values()], axis=0)
feature_names = list(list(feature_dict.values())[0].keys())

if 1:
  # (1) For categorical targets:
  targets = [0 if g == 'M' else 1 for g in fake_genders]
  target_labels = ['Male', 'Female']
else:
  # (2) For numerical targets:
  targets = fake_ages
  target_labels = ['Age']

omix = Omix(features, targets, feature_names, None, target_labels,
            data_name=f'SC-153x375-{TIME_RESOLUTION}s')
omix.show_in_explorer()

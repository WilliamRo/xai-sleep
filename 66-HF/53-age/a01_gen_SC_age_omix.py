# Add path in order to be compatible with Linux
import sys, os

SOLUTION_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

PATH_LIST = ['32-SC']

print(f'[66-53] Solution dir = {SOLUTION_DIR}')
sys.path.append(SOLUTION_DIR)
for p in PATH_LIST: sys.path.append(os.path.join(SOLUTION_DIR, p))

import sc as hub
# -----------------------------------------------------------------------------
from hypnomics.hypnoprints.extractor import Extractor
from pictor.xomics.omix import Omix
from roma import finder

import numpy as np



# -----------------------------------------------------------------------------
# (1) Configuration
# -----------------------------------------------------------------------------
# (1.1) Nebula configuration
SG_LABELS = finder.walk(hub.CLOUD_DIR, type_filter='dir', return_basename=True)

# [2, 5, 10, 30]
TIME_RESOLUTION = 2
if len(sys.argv) == 2: TIME_RESOLUTION = int(sys.argv[1])
assert TIME_RESOLUTION in [2, 5, 10, 30]

CHANNELS = ['EEG Fpz-Cz', 'EEG Pz-Oz']

# (1.2) TODO: Configure here !
CONDITIONAL = 1
PROBE_CONFIG = 'Ab'
INCLUDE_WAKE = 0
OVERWRITE = 0

NEB_FN = hub.get_neb_file_name(TIME_RESOLUTION, PROBE_CONFIG)

# (1.4) MISC
C_SUFFIX = f'{"c" if CONDITIONAL else "nc"}'
W_SUFFIX = '' if INCLUDE_WAKE else '-NW'

PROBE_KEYS = hub.probe_tools.get_probe_keys(PROBE_CONFIG, expand_group=True)
PROBE_SUFFIX = hub.probe_tools.get_probe_suffix(PROBE_CONFIG)

# -----------------------------------------------------------------------------
# (2) Load Omix
# -----------------------------------------------------------------------------
OMIX_FN = f'SC-{TIME_RESOLUTION}s-{PROBE_SUFFIX}-{C_SUFFIX}{W_SUFFIX}.omix'
OMIX_PATH = os.path.join(hub.OMIX_DIR, OMIX_FN)

if os.path.exists(OMIX_PATH) and not OVERWRITE:
  omix = Omix.load(OMIX_PATH)
else:
  from hypnomics.freud.nebula import Nebula

  # (3.1) Load Nebula
  nebula: Nebula = Nebula.load(os.path.join(hub.NEBULA_DIR, NEB_FN))
  nebula.set_probe_keys(PROBE_KEYS)

  # (2.2) Extract features
  if CONDITIONAL:
    E_SETTINGS = {
      'include_statistical_features': 1,
      'include_inter_stage_features': 1,
      'include_inter_channel_features': 1,
    }
  else:
    E_SETTINGS = {
      'include_statistical_features': 0,
      'include_inter_stage_features': 0,
      'include_inter_channel_features': 0,
      'include_all_mean_std': 1,
    }

  extractor = Extractor(**E_SETTINGS)
  feature_dict = extractor.extract(nebula, return_dict=True)
  features = np.stack([np.array(list(v.values()))
                       for v in feature_dict.values()], axis=0)
  feature_names = list(list(feature_dict.values())[0].keys())

  target_labels = ['Age']
  meta = hub.sc_tools.load_sc_meta(hub.XLSX_PATH, nebula.labels)
  targets = [meta[pid]['age'] for pid in nebula.labels]

  data_name = OMIX_FN.split('.')[0]
  omix = Omix(features, targets, feature_names, nebula.labels, target_labels,
              data_name=f'SC-age-153-{TIME_RESOLUTION}s')

  if not INCLUDE_WAKE: omix = omix.filter_by_name('W', include=False)

  # (2.3) Save omix
  omix.save(OMIX_PATH)

# -----------------------------------------------------------------------------
# (4) Visualization
# -----------------------------------------------------------------------------
omix.show_in_explorer()

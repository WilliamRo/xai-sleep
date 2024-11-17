from hf.probe_tools import get_probe_keys
from hypnomics.hypnoprints.extractor import Extractor
from pictor.xomics.omix import Omix
from roma import finder

import numpy as np
import os



# -----------------------------------------------------------------------------
# (1) Configuration
# -----------------------------------------------------------------------------
# (1.1) Nebula configuration
WORK_DIR = r'../data/sleepedfx_sc'
SG_LABELS = finder.walk(WORK_DIR, type_filter='dir', return_basename=True)
NEB_FN = f'SC-30s-ABC38.nebula'
MACRO_PATH = r'P:\xai-sleep\66-HF\03-sleep-age\data\SC-age-macro-30.omix'

# [ 2(x), 5(x), 10(x), 30, ]
TIME_RESOLUTION = 30

CHANNELS = [
  'EEG Fpz-Cz',
  'EEG Pz-Oz'
]

# (1.2) TODO: Configure here !
CONDITIONAL = 1
PROBE_CONFIG = 'ABD'
INCLUDE_MACRO = 1
INCLUDE_WAKE = 0
OVERWRITE = 1

# (1.4) MISC
C_SUFFIX = f'{"c" if CONDITIONAL else "nc"}'
W_SUFFIX = '' if INCLUDE_WAKE else '-NW'

PROBE_KEYS = get_probe_keys(PROBE_CONFIG)
PROBE_SUFFIX = f'{PROBE_CONFIG}{len(PROBE_KEYS)}'
MACRO_SUFFIX = f'-MACRO' if INCLUDE_MACRO else ''

OMIX_FN = f'SC-30s-{PROBE_SUFFIX}-{C_SUFFIX}{MACRO_SUFFIX}{W_SUFFIX}.omix'
# -----------------------------------------------------------------------------
# (2) Load macro omix
# -----------------------------------------------------------------------------
macro_omix = Omix.load(MACRO_PATH)

# -----------------------------------------------------------------------------
# (3) Load Omix
# -----------------------------------------------------------------------------
OMIX_PATH = os.path.join(WORK_DIR, OMIX_FN)
if os.path.exists(OMIX_PATH) and not OVERWRITE:
  omix = Omix.load(OMIX_PATH)
else:
  from hypnomics.freud.nebula import Nebula

  # (3.1) Load Nebula
  nebula: Nebula = Nebula.load(os.path.join(WORK_DIR, NEB_FN))
  nebula.set_probe_keys(PROBE_KEYS)

  # (2.2) Extract features
  if CONDITIONAL:
    E_SETTINGS = {
      'include_statistical_features': 1,
      'include_inter_stage_features': 1,
      'include_inter_channel_features': 1,

      'include_proportion': 0,
      'include_stage_shift': 0,
      'include_channel_shift': 0,
      'include_stage_wise_covariance': 0,
      'include_stage_mean': 0,
      'include_all_mean_std': 0,
    }
  else:
    E_SETTINGS = {
      'include_proportion': 0,
      'include_stage_shift': 0,
      'include_channel_shift': 0,
      'include_stage_wise_covariance': 0,
      'include_stage_mean': 0,
      'include_all_mean_std': 1,
    }

  extractor = Extractor(**E_SETTINGS)
  feature_dict = extractor.extract(nebula, return_dict=True)
  features = np.stack([np.array(list(v.values()))
                       for v in feature_dict.values()], axis=0)
  feature_names = list(list(feature_dict.values())[0].keys())

  target_labels = ['Age']
  targets = [nebula.meta[pid]['age'] for pid in nebula.labels]

  data_name = OMIX_FN.split('.')[0]
  omix = Omix(features, targets, feature_names, nebula.labels, target_labels,
              data_name=f'SC-age-153-{TIME_RESOLUTION}s')

  if INCLUDE_MACRO: omix = omix * macro_omix
  if not INCLUDE_WAKE: omix = omix.filter_by_name('W', include=False)

  # (2.3) Save omix
  omix.save(OMIX_PATH)

# -----------------------------------------------------------------------------
# (4) Visualization
# -----------------------------------------------------------------------------
omix.show_in_explorer()

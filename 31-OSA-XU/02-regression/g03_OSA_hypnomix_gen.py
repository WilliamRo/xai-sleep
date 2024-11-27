from hf.probe_tools import get_probe_keys
from hypnomics.freud.nebula import Nebula
from hypnomics.hypnoprints.extractor import Extractor
from osaxu.osa_tools import set_target_collection_for_omix
from pictor.xomics.omix import Omix
from roma import finder

import numpy as np
import os



# -----------------------------------------------------------------------------
# (1) Configuration
# -----------------------------------------------------------------------------
# (1.1) Nebula configuration
WORK_DIR = r'../data'  # contains cloud files
SG_LABELS = finder.walk(WORK_DIR, type_filter='dir', return_basename=True,
                        pattern='*')

# (1.2) Nebula path
NEB_FN = f'125samples-6channels-39probes-30s.nebula'
NEB_PATH = os.path.join(WORK_DIR, NEB_FN)

# (1.3) Set default target
SUN = 1
SUN_SUFFIX = '-SUN' if SUN else ''
if SUN:
  PROBE_CONFIG = 'C'
  INCLUDE_WAKE = 1
else:
  PROBE_CONFIG = 'ABD'
  INCLUDE_WAKE = 0

TARGET = 'age'
PROBE_KEYS = get_probe_keys(PROBE_CONFIG)

W_SUFFIX = 'W' if INCLUDE_WAKE else 'NW'
OVERWRITE = 0
# -----------------------------------------------------------------------------
# (2)
# -----------------------------------------------------------------------------
OMIX_FN = f'125samples-6channels-{PROBE_CONFIG}-30s-{W_SUFFIX}{SUN_SUFFIX}-{TARGET}.omix'
OMIX_PATH = os.path.join(WORK_DIR, OMIX_FN)

if os.path.exists(OMIX_PATH) and not OVERWRITE:
  omix = Omix.load(OMIX_PATH)
else:
  nebula = Nebula.load(NEB_PATH)
  nebula.set_probe_keys(PROBE_KEYS)

  if SUN:
    E_SETTINGS = {
      'include_statistical_features': 0,
      'include_inter_stage_features': 0,
      'include_inter_channel_features': 0,

      'include_proportion': 0,
      'include_stage_shift': 0,
      'include_channel_shift': 0,
      'include_stage_wise_covariance': 0,
      'include_stage_mean': 1,
      'include_all_mean_std': 0,
    }
  else:
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

  extractor = Extractor(**E_SETTINGS)
  feature_dict = extractor.extract(nebula, return_dict=True)
  features = np.stack([np.array(list(v.values()))
                       for v in feature_dict.values()], axis=0)
  feature_names = list(list(feature_dict.values())[0].keys())

  target_labels = [TARGET]
  targets = [nebula.meta[pid][TARGET] for pid in nebula.labels]
  omix = Omix(features, targets, feature_names, nebula.labels, target_labels,
              data_name=OMIX_FN)

  set_target_collection_for_omix(omix, nebula)

  if not INCLUDE_WAKE: omix = omix.filter_by_name('W', include=False)
  omix.save(OMIX_PATH)

omix.show_in_explorer()

"""This script plots age-distance figure.
"""
# Add path in order to be compatible with Linux
import sys, os

SOLUTION_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

PATH_LIST = ['26-HSP', '26-HSP/99-data-probe', 'xai-kit', 'hypnomics',
             'xai-kit/roma', 'xai-kit/pictor', 'xai-kit/tframe']

if __name__ == '__main__':
  print(f'Solution dir = {SOLUTION_DIR}')
  sys.path.append(SOLUTION_DIR)
  for p in PATH_LIST: sys.path.append(os.path.join(SOLUTION_DIR, p))

from hf.dist_agent import DistanceAgent
from hf.probe_tools import get_probe_keys
from roma import io

import a00_common as hub



# -----------------------------------------------------------------------------
# (0) Configuration
# -----------------------------------------------------------------------------
# TODO: configure here
subset_dict_fn = hub.SubsetDicts.ss_2ses_3types_378
subset_dict_path = os.path.join(SOLUTION_DIR, 'data/hsp', subset_dict_fn)

DA_KEY = f'HSP378'

TIME_RESOLUTION = 30
CHANNELS = ['EEG F3-M2', 'EEG C3-M2', 'EEG O1-M2',
            'EEG F4-M1', 'EEG C4-M1', 'EEG O2-M1']
PROBE_CONFIG = 'Ad'
CONDITIONAL = [0, 1]
SHIFT_COMPENSATION = [0, 1]

PROBE_KEYS = get_probe_keys(PROBE_CONFIG, expand_group=True)

# -----------------------------------------------------------------------------
# (1) Load subset dict and pairs
# -----------------------------------------------------------------------------
assert os.path.exists(subset_dict_path)
patient_dict = io.load_file(subset_dict_path, verbose=True)

N = sum([len(v) for v in patient_dict.values()])
hub.console.show_status(
  f'There are {len(patient_dict)} patients with at least 2 sessions with '
  f'annotation, altogether {N} folders.')

label_pairs, ad_dict = hub.ha.get_longitudinal_pairs(patient_dict, True)
# -----------------------------------------------------------------------------
# (2) Load distance
# -----------------------------------------------------------------------------
da = DistanceAgent(DA_KEY, hub.DIST_DIR, hub.CLOUD_DIR)

# -----------------------------------------------------------------------------
# (3) Plot figure
# -----------------------------------------------------------------------------
da.plot_age_delta_distance(TIME_RESOLUTION, CHANNELS, PROBE_KEYS,
                           label_pairs, ad_dict, figsize=(10, 6),
                           cond=0, comp=0, patient_dict=patient_dict)

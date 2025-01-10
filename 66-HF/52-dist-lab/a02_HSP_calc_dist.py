# Add path in order to be compatible with Linux
import sys, os

SOLUTION_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

PATH_LIST = ['26-HSP', '26-HSP/99-data-probe', 'xai-kit', 'hypnomics', '66-HF',
             'xai-kit/roma', 'xai-kit/pictor', 'xai-kit/tframe']

if __name__ == '__main__':
  print(f'Solution dir = {SOLUTION_DIR}')
  sys.path.append(SOLUTION_DIR)
  for p in PATH_LIST: sys.path.append(os.path.join(SOLUTION_DIR, p))
# -----------------------------------------------------------------------------

from hf.dist_agent import DistanceAgent
from hf.probe_tools import get_probe_keys

import a00_common as hub



# -----------------------------------------------------------------------------
# (1) Configuration
# -----------------------------------------------------------------------------
WORK_DIR = os.path.join(SOLUTION_DIR, 'data/hsp/hsp_dist')
SUBSET_FN = hub.SubsetDicts.ss_2ses_3types_378

N_SUBJECTS = 50
MAD = 0.2

OVERWRITE = 0

TIME_RESOLUTION = 30
CHANNELS = ['EEG F3-M2', 'EEG C3-M2', 'EEG O1-M2',
            'EEG F4-M1', 'EEG C4-M1', 'EEG O2-M1']
PROBE_CONFIG = 'Ad'
CONDITIONAL = [0, 1]
SHIFT_COMPENSATION = [0, 1]

PROBE_KEYS = get_probe_keys(PROBE_CONFIG, expand_group=True)
# -----------------------------------------------------------------------------
# (2) Get subset pairs
# -----------------------------------------------------------------------------
# folder_list = hub.ha.load_subset_dict(SUBSET_FN, return_folder_list=True)
labels_n1, labels_n2 = hub.ha.load_pair_labels(SUBSET_FN)
labels_n1, labels_n2 = labels_n1[:N_SUBJECTS], labels_n2[:N_SUBJECTS]

# -----------------------------------------------------------------------------
# (3) Calculate distance
# -----------------------------------------------------------------------------
DA_KEY = f'HSP378'
da = DistanceAgent(DA_KEY, WORK_DIR, hub.CLOUD_DIR, labels_n1, labels_n2)

for conditional in CONDITIONAL:
  for compensate_shift in SHIFT_COMPENSATION:
    da.calculate_distance(
      time_resolution=TIME_RESOLUTION,
      channels=CHANNELS,
      probe_keys=PROBE_KEYS,
      conditional=conditional,
      overwrite=OVERWRITE,
      compensate_shift=compensate_shift,
    )

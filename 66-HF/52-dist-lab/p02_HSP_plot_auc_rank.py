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
from roma import console, finder

import a00_common as hub



# -----------------------------------------------------------------------------
# (1) Configuration
# -----------------------------------------------------------------------------
WORK_DIR = hub.DIST_DIR
SUBSET_FN = hub.SubsetDicts.ss_2ses_3types_378

N_SUBJECTS = 50
# MAD = 0.2

TIME_RESOLUTION = 30
CHANNELS = ['EEG F3-M2', 'EEG F4-M1', 'EEG C3-M2',
            'EEG C4-M1', 'EEG O1-M2', 'EEG O2-M1']
# CHANNELS = ['EEG F3-M2', 'EEG C3-M2']
PROBE_CONFIG = 'Ad'

PROBE_KEYS = get_probe_keys(PROBE_CONFIG, expand_group=True)
PROBE_KEYS.remove('KURT')
# -----------------------------------------------------------------------------
# (2) Get subset pairs
# -----------------------------------------------------------------------------
labels_n1, labels_n2 = hub.ha.load_pair_labels(SUBSET_FN)
labels_n1, labels_n2 = labels_n1[:N_SUBJECTS], labels_n2[:N_SUBJECTS]

# -----------------------------------------------------------------------------
# (3)
# -----------------------------------------------------------------------------
DA_KEY = f'HSP378'
da = DistanceAgent(DA_KEY, WORK_DIR, hub.CLOUD_DIR, labels_n1, labels_n2)

da.plot_auc_rank(CHANNELS, PROBE_KEYS, TIME_RESOLUTION, figsize=(7, 9))

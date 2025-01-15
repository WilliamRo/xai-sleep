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

import hf.sc_tools as sc



# -----------------------------------------------------------------------------
# (1) Configuration
# -----------------------------------------------------------------------------
WORK_DIR = os.path.join(SOLUTION_DIR, 'data/sleepedfx-sc/sc_dist')
CLOUD_DIR = os.path.join(SOLUTION_DIR, 'data/sleepedfx-sc/sc_clouds')

N_SUBJECTS = 75

TIME_RESOLUTION = 2
CHANNELS = ['EEG Fpz-Cz', 'EEG Pz-Oz']
PROBE_CONFIG = 'Ad'

PROBE_KEYS = get_probe_keys(PROBE_CONFIG, expand_group=True)
# -----------------------------------------------------------------------------
# (2) Get subset pairjks
# -----------------------------------------------------------------------------
sg_labels = finder.walk(CLOUD_DIR, 'dir', return_basename=True)
labels_n1, labels_n2 = sc.get_paired_sg_labels(sg_labels, return_two_lists=True)
labels_n1, labels_n2 = labels_n1[:N_SUBJECTS], labels_n2[:N_SUBJECTS]

# -----------------------------------------------------------------------------
# (3)
# -----------------------------------------------------------------------------
DA_KEY = f'SC75'
da = DistanceAgent(DA_KEY, WORK_DIR, CLOUD_DIR, labels_n1, labels_n2)

da.plot_auc_rank(CHANNELS, PROBE_KEYS, TIME_RESOLUTION)
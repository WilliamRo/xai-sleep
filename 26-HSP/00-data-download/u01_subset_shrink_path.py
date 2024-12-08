# Add path in order to be compatible with Linux
import sys, os

SOLUTION_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
print(f'Solution dir = {SOLUTION_DIR}')

PATH_LIST = ['26-HSP', '26-HSP/01-sg-conversion', 'xai-kit',
             'xai-kit/roma', 'xai-kit/pictor', 'xai-kit/tframe']

sys.path.append(SOLUTION_DIR)
for p in PATH_LIST: sys.path.append(os.path.join(SOLUTION_DIR, p))

from freud.talos_utils.sleep_sets.hsp import HSPAgent
from roma.console.console import console

import numpy as np



# -----------------------------------------------------------------------------
# (1) Configuration
# -----------------------------------------------------------------------------
if os.name == 'nt':
  console.show_status('Windows system detected.')
  DATA_DIR = os.path.join(SOLUTION_DIR, r'\\192.168.5.100\xai-beta\xai-sleep\data\hsp\hsp_raw')
else:
  console.show_status('Linux system detected.')
  DATA_DIR = os.path.join(SOLUTION_DIR, 'data/hsp/hsp_raw')

META_DIR = os.path.join(SOLUTION_DIR, 'data/hsp')
SG_DIR = os.path.join(SOLUTION_DIR, 'data/hsp/hsp_sg')

META_TIME_STAMP = '20231101'
META_PATH = os.path.join(
  META_DIR, DATA_DIR, f'bdsp_psg_master_{META_TIME_STAMP}.csv')

# -----------------------------------------------------------------------------
# (2) Stage I: include subjects with at least 2 sessions with stage annotation
#              according to the meta data
# -----------------------------------------------------------------------------
ha = HSPAgent(META_DIR, DATA_DIR, META_TIME_STAMP)
console.show_status(f'Total subject number: {len(ha.patient_dict)}',
                    prompt='[META]')

patient_dict = ha.filter_patients_meta(
  min_n_sessions=2, should_have_annotation=True)
folder_list = ha.convert_to_folder_names(patient_dict)

console.show_status(
  f'There are {len(patient_dict)} patients with at least 2 sessions with '
  f'annotation, altogether {len(folder_list)} folders.', prompt='[META]')

# -----------------------------------------------------------------------------
# (3) Stage II: exclude
# -----------------------------------------------------------------------------
patient_dict = ha.filter_patients_local(
  patient_dict, min_n_sessions=2, should_have_annotation=True, verbose=True)
folder_list = ha.convert_to_folder_names(patient_dict)
console.show_status(
  f'There are {len(patient_dict)} patients with at least 2 sessions with '
  f'annotation, altogether {len(folder_list)} folders.', prompt='[LOCAL]')

# -----------------------------------------------------------------------------
# (4) Stage III: sg filter
# -----------------------------------------------------------------------------
patient_dict = ha.filter_patients_sg(
  patient_dict, SG_DIR, min_n_sessions=2, dtype=np.float16, max_sfreq=128,
  verbose=True)
folder_list = ha.convert_to_folder_names(patient_dict)
console.show_status(
  f'There are {len(patient_dict)} patients with at least 2 sessions with '
  f'annotation, altogether {len(folder_list)} folders.', prompt='[SG]')

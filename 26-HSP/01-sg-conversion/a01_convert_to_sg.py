# Add path in order to be compatible with Linux
import sys, os

SOLUTION_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
print(f'Solution dir = {SOLUTION_DIR}')

PATH_LIST = ['26-HSP', '26-HSP/01-sg-conversion', 'xai-kit',
             'xai-kit/roma', 'xai-kit/pictor', 'xai-kit/tframe']

for p in PATH_LIST: sys.path.append(os.path.join(SOLUTION_DIR, p))

# Import anything here
from freud.talos_utils.sleep_sets.hps import HSPAgent, HSPSet
from roma import console



# -----------------------------------------------------------------------------
# (1) Configuration
# -----------------------------------------------------------------------------
SRC_PATH = r'F:\data\hsp'
TGT_PATH = os.path.join(SOLUTION_DIR, '26-HSP/data/hsp_sg')

META_DIR = r'E:\xai-sleep\data\hsp'
META_TIME_STAMP = '20231101'

# -----------------------------------------------------------------------------
# (2) Conversion
# -----------------------------------------------------------------------------
ha = HSPAgent(META_DIR, data_dir=None, meta_time_stamp=META_TIME_STAMP)

patient_dict = ha.filter_patients(min_n_sessions=2, should_have_annotation=1)
folder_list = ha.convert_to_folder_names(patient_dict)

console.show_status(f'{len(folder_list)} .edf files found in `{SRC_PATH}`')


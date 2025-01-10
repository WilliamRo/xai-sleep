# Add path in order to be compatible with Linux
import sys, os

SOLUTION_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
print(f'Solution dir = {SOLUTION_DIR}')

PATH_LIST = ['26-HSP', '26-HSP/99-data-probe', 'xai-kit', 'hypnomics',
             'xai-kit/roma', 'xai-kit/pictor', 'xai-kit/tframe']

sys.path.append(SOLUTION_DIR)
for p in PATH_LIST: sys.path.append(os.path.join(SOLUTION_DIR, p))

from freud.talos_utils.sleep_sets.hsp import HSPSet
from roma import console

import a00_common as hub



# -----------------------------------------------------------------------------
# (1) Configuration
# -----------------------------------------------------------------------------
SUBSET_FN = hub.SubsetDicts.ss_2ses_3types_bipolar_218
TGT_PATH = os.path.join(SOLUTION_DIR, 'data/hsp/hsp_sg_bipolar')


# -----------------------------------------------------------------------------
# (2) Conversion
# -----------------------------------------------------------------------------
folder_list = hub.ha.load_subset_dict(SUBSET_FN, return_folder_list=True)

console.show_status(f'{len(folder_list)} .edf files should be converted.')

success_path_list = HSPSet.convert_rawdata_to_signal_groups(
  ses_folder_list=folder_list, tgt_dir=TGT_PATH, bipolar=True, overwrite=1)

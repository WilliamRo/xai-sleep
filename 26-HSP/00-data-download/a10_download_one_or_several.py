# Add path in order to be compatible with Linux
import sys, os

SOLUTION_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
print(f'Solution dir = {SOLUTION_DIR}')

PATH_LIST = ['26-HSP', '26-HSP/01-sg-conversion', 'xai-kit',
             'xai-kit/roma', 'xai-kit/pictor', 'xai-kit/tframe']

sys.path.append(SOLUTION_DIR)
for p in PATH_LIST: sys.path.append(os.path.join(SOLUTION_DIR, p))

import os
import sys


from freud.talos_utils.sleep_sets.hsp import HSPAgent
from roma.console.console import console

# -----------------------------------------------------------------------------
# (1) Configuration
# -----------------------------------------------------------------------------
ACCESS_POINT_NAME = 's3://arn:aws:s3:us-east-1:184438910517:accesspoint'
DATA_DIR = os.path.join(SOLUTION_DIR, 'data/hsp/hsp_raw')
META_DIR = os.path.join(SOLUTION_DIR, 'data/hsp')

META_TIME_STAMP = '20231101'
META_PATH = os.path.join(
  META_DIR, DATA_DIR, f'bdsp_psg_master_{META_TIME_STAMP}.csv')

# -----------------------------------------------------------------------------
# (2) Select folders
# -----------------------------------------------------------------------------
ha = HSPAgent(META_DIR, DATA_DIR, META_TIME_STAMP, ACCESS_POINT_NAME)
folder_list = ha.convert_to_folder_names(ha.patient_dict)

# BETA
# sub-S0001120608302/ses-2: META.psq = 'Y' but not exist
folder_list = [fn for fn in folder_list if 'sub-S0001120608302/ses-2' in fn]
assert len(folder_list) == 1

ha.download_folders(folder_list)




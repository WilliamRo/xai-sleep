import os
import sys


from freud.talos_utils.sleep_sets.hps import HSPAgent
from roma.console.console import console

# -----------------------------------------------------------------------------
# (1) Configuration
# -----------------------------------------------------------------------------
ACCESS_POINT_NAME = 's3://arn:aws:s3:us-east-1:[5f5s51]438910517:accesspoint'
ACCESS_POINT_NAME = 's3://arn:aws:s3:us-east-1:184438910517:accesspoint'
DATA_DIR = '../../data/hsp'
META_DIR = '../../data/hsp'

META_TIME_STAMP = '20231101'
META_PATH = os.path.join(
  META_DIR, DATA_DIR, f'bdsp_psg_master_{META_TIME_STAMP}.csv')

# -----------------------------------------------------------------------------
# (2) Select folders
# -----------------------------------------------------------------------------
ha = HSPAgent(META_DIR, DATA_DIR, META_TIME_STAMP, ACCESS_POINT_NAME)
patient_dict = ha.filter_patients(min_n_sessions=2)
folder_list = ha.convert_to_folder_names(patient_dict)

console.show_status(f'There are {len(patient_dict)} patients with at least 2 sessions.')

ha.download_folders(folder_list[:10])




# Add path in order to be compatible with Linux
import sys, os

SOLUTION_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
print(f'Solution dir = {SOLUTION_DIR}')

PATH_LIST = ['26-HSP', '26-HSP/01-sg-conversion', 'xai-kit',
             'xai-kit/roma', 'xai-kit/pictor', 'xai-kit/tframe']

sys.path.append(SOLUTION_DIR)
for p in PATH_LIST: sys.path.append(os.path.join(SOLUTION_DIR, p))

# Import anything here
from freud.talos_utils.sleep_sets.hsp import HSPAgent, HSPSet
from roma import console



# -----------------------------------------------------------------------------
# (1) Configuration
# -----------------------------------------------------------------------------
SRC_PATH = os.path.join(SOLUTION_DIR, r'\\192.168.5.100\xai-beta\xai-sleep\data\hsp\hsp_raw')
TGT_PATH = os.path.join(SOLUTION_DIR, 'data/hsp/hsp_sg')

META_DIR = os.path.join(SOLUTION_DIR, 'data/hsp')
META_TIME_STAMP = '20231101'

# -----------------------------------------------------------------------------
# (2) Conversion
# -----------------------------------------------------------------------------
ha = HSPAgent(META_DIR, data_dir=SRC_PATH, meta_time_stamp=META_TIME_STAMP)

patient_dict = ha.filter_patients_meta(min_n_sessions=2, should_have_annotation=1)
folder_list = ha.convert_to_folder_names(patient_dict, local=True)

PROBAMETIC_ID = [
  'sub-S0001121829856/ses-2',  # no standard annotation
  'sub-S0001121493813/ses-1',  # no standard annotation
  'sub-S0001117897144/ses-2',  # no standard annotation
  'sub-S0001117585804/ses-2',  # no stage annotation (marked yes in META)
][0]

folder_list = [fn for fn in folder_list if PROBAMETIC_ID in fn]

console.show_status(f'{len(folder_list)} .edf files should be converted.')

sg_list = HSPSet.convert_rawdata_to_signal_groups(
  ses_folder_list=folder_list, tgt_dir=TGT_PATH)

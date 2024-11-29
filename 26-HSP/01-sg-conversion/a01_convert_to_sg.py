# Add path in order to be compatible with Linux
import sys, os

SOLUTION_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
print(f'Solution dir = {SOLUTION_DIR}')

PATH_LIST = ['26-HSP', '26-HSP/01-sg-conversion', 'xai-kit',
             'xai-kit/roma', 'xai-kit/pictor', 'xai-kit/tframe']

for p in PATH_LIST: sys.path.append(os.path.join(SOLUTION_DIR, p))

# Import anything here
from freud.gui.freud_gui import Freud
from freud.talos_utils.sleep_sets.hsp import HSPAgent, HSPSet
from roma import console
from pictor.objects import SignalGroup



# -----------------------------------------------------------------------------
# (1) Configuration
# -----------------------------------------------------------------------------
SRC_PATH = os.path.join(SOLUTION_DIR, 'data/hsp/hsp_raw')
TGT_PATH = os.path.join(SOLUTION_DIR, 'data/hsp/hsp_sg')

META_DIR = os.path.join(SOLUTION_DIR, 'data/hsp')
META_TIME_STAMP = '20231101'

# -----------------------------------------------------------------------------
# (2) Conversion
# -----------------------------------------------------------------------------
ha = HSPAgent(META_DIR, data_dir=SRC_PATH, meta_time_stamp=META_TIME_STAMP)

patient_dict = ha.filter_patients(min_n_sessions=2, should_have_annotation=1)
folder_list = ha.convert_to_folder_names(patient_dict, local=True)

console.show_status(f'{len(folder_list)} .edf files should be converted.')

sg: SignalGroup = HSPSet.load_sg_from_raw_files(ses_dir=folder_list[0])

Freud.visualize_signal_groups(
  [sg],
  title='HSP', default_win_duration=9999999,
)

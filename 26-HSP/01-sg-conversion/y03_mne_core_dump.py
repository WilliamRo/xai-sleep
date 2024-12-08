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

# For 'Segmentation fault (core dumped)' error
import faulthandler
faulthandler.enable()

# Handle numba error
# os.environ['NUMBA_DISABLE_INTEL_SVML'] = '1'  # not work


# -----------------------------------------------------------------------------
# (1) Configuration
# -----------------------------------------------------------------------------
if os.name == 'nt': SOLUTION_DIR = r'\\192.168.5.100\xai-beta\xai-sleep'

SRC_PATH = os.path.join(SOLUTION_DIR, 'data/hsp/hsp_raw')
TGT_PATH = os.path.join(SOLUTION_DIR, 'data/hsp/hsp_sg')

META_DIR = os.path.join(SOLUTION_DIR, 'data/hsp')
META_TIME_STAMP = '20231101'

# -----------------------------------------------------------------------------
# (2) Generate folder list
# -----------------------------------------------------------------------------
ha = HSPAgent(META_DIR, data_dir=SRC_PATH, meta_time_stamp=META_TIME_STAMP)

patient_dict = ha.filter_patients_meta(min_n_sessions=2, should_have_annotation=1)
patient_dict = ha.filter_patients_local(patient_dict, min_n_sessions=2,
                                        should_have_annotation=1)

folder_list = ha.convert_to_folder_names(patient_dict, local=True)

console.show_status(f'{len(folder_list)} .edf files should be converted.')

# -----------------------------------------------------------------------------
# (3)
# -----------------------------------------------------------------------------
from pictor.objects.signals.signal_group import DigitalSignal, SignalGroup

ses_path = os.path.join(SRC_PATH, 'sub-S0001111230101/ses-2')
sg: SignalGroup = HSPSet.load_sg_from_raw_files(ses_dir=ses_path)

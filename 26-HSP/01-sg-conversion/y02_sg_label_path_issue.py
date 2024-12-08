# Add path in order to be compatible with Linux
import sys, os

SOLUTION_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
print(f'Solution dir = {SOLUTION_DIR}')

PATH_LIST = ['26-HSP', '26-HSP/01-sg-conversion', 'xai-kit',
             'xai-kit/roma', 'xai-kit/pictor', 'xai-kit/tframe']

sys.path.append(SOLUTION_DIR)
for p in PATH_LIST: sys.path.append(os.path.join(SOLUTION_DIR, p))

# Import anything here
from freud.gui.freud_gui import Freud
from freud.talos_utils.sleep_sets.hsp import HSPAgent, HSPSet, HSPOrganization
from roma import io



# -----------------------------------------------------------------------------
# (1) Configuration
# -----------------------------------------------------------------------------
SOLUTION_DIR = r'\\192.168.5.100\xai-beta\xai-sleep'
SRC_PATH = os.path.join(SOLUTION_DIR, 'data/hsp/hsp_raw')
TGT_PATH = os.path.join(SOLUTION_DIR, 'data/hsp/hsp_sg')

META_DIR = os.path.join(SOLUTION_DIR, 'data/hsp')
META_TIME_STAMP = '20231101'

# -----------------------------------------------------------------------------
# (2) Get folder list
# -----------------------------------------------------------------------------
signal_groups = []

# a
sub_id = 'sub-s0001111326132'
ses_id = 'ses-4'

sg_path = os.path.join(TGT_PATH, f'{sub_id}_{ses_id}(float16,128Hz).sg')
sg_in_disk = io.load_file(sg_path, verbose=True)
signal_groups.append(sg_in_disk)

# b
miss_sub_id = 'sub-s0001111321859'
miss_ses_id = 'ses-2'

miss_sg_path = os.path.join(TGT_PATH, f'{miss_sub_id}_{miss_ses_id}(float16,128Hz).sg')
miss_sg_in_disk = io.load_file(miss_sg_path, verbose=True)
signal_groups.append(miss_sg_in_disk)

# c
ses_path = os.path.join(SRC_PATH, f'{sub_id}/{ses_id}')
correct_sg = HSPSet.load_sg_from_raw_files(ses_dir=ses_path)

# (x) Visualization
Freud.visualize_signal_groups(
  signal_groups, title='HSP', default_win_duration=9999999,
)

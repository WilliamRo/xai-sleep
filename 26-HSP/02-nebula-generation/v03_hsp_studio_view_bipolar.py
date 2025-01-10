# Add path in order to be compatible with Linux
import sys, os

SOLUTION_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

PATH_LIST = ['26-HSP', '26-HSP/99-data-probe', 'xai-kit', 'hypnomics',
             'xai-kit/roma', 'xai-kit/pictor', 'xai-kit/tframe']

if __name__ == '__main__':
  print(f'Solution dir = {SOLUTION_DIR}')
  sys.path.append(SOLUTION_DIR)
  for p in PATH_LIST: sys.path.append(os.path.join(SOLUTION_DIR, p))

from a00_common import ha, console, SubsetDicts, SG_DIR, CLOUD_DIR, IN_LINUX
from hypnomics.freud import HypnoStudio
from roma import console, finder, io



# -----------------------------------------------------------------------------
# (1) Configuration
# -----------------------------------------------------------------------------
TIME_RESOLUTION = 30
CHANNELS = ['EEG Fpz-Cz', 'EEG Pz-Oz']
PROBE_KEYS = [
  'FREQ-20',   # 0
  'AMP-1',     # 1
  'RP-DELTA',  # 3
  'RP-THETA',  # 4
  'RP-ALPHA',  # 5
  'RP-BETA',   # 6
  'PR-DELTA_TOTAL',  # 7
  'PR-THETA_TOTAL',  # 8
  'PR-ALPHA_TOTAL',  # 9
]
PK1 = PROBE_KEYS[0]
PK2 = PROBE_KEYS[8]

STUDIO_DIR = os.path.join(SOLUTION_DIR, 'data/hsp/hsp_studio')
# -----------------------------------------------------------------------------
# (2) Construct a HypnoStudio
# -----------------------------------------------------------------------------
hs = HypnoStudio(work_dir=STUDIO_DIR, sg_dir=SG_DIR, neb_dir=CLOUD_DIR)

# -----------------------------------------------------------------------------
# (3) Generate a testing photo
# -----------------------------------------------------------------------------
SG_DIR += '_bipolar'
sg_pattern = f'sub-S0001111236530_ses-1(float16,128Hz,bipolar).sg'

sg_file_list = finder.walk(SG_DIR, pattern=sg_pattern)
sg_path = sg_file_list[0]

neb_fn = os.path.basename(sg_path).split('(')[0]
neb_path = os.path.join(CLOUD_DIR, neb_fn)

hs.take_one_photo(sg_path, neb_path, CHANNELS, TIME_RESOLUTION, PK1, PK2,
                  show_figure=True, properties={}, fig_size=(9, 8),
                  save=True)


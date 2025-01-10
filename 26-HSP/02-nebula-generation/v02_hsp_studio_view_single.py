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
CHANNELS = ['EEG F3-M2', 'EEG C3-M2', 'EEG O1-M2',
            'EEG F4-M1', 'EEG C4-M1', 'EEG O2-M1']
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
# sg_pattern = f'sub-S*111239185_ses-2(float16,128Hz).sg'
sg_pattern = f'sub-S0001111236530_ses-2(float16,128Hz).sg'

sg_file_list = finder.walk(SG_DIR, pattern=sg_pattern)
sg_path = sg_file_list[0]

neb_fn = os.path.basename(sg_path).split('(')[0]
neb_path = os.path.join(CLOUD_DIR, neb_fn)

hs.take_one_photo(sg_path, neb_path, CHANNELS, TIME_RESOLUTION, PK1, PK2,
                  show_figure=not IN_LINUX, properties={}, fig_size=(9, 8),
                  save=True)


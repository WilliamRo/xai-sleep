from hypnomics.freud import HypnoStudio
from hypnomics.freud.nebula import Nebula
from roma import console, finder, io

import os



# -----------------------------------------------------------------------------
# (1) Configuration
# -----------------------------------------------------------------------------
# (1.1) Working directory & patient inclusion
SG_DIR = r'../../data/sleep-edf-database-expanded-1.0.0/sleep-cassette'
SG_PATTERN = f'*(trim1800;128).sg'

NEB_DIR = r'../data/sleepedfx_sc'

STUDIO_DIR = r'../../data/sleepedfx-sc/sc_studio'

# (1.2) Photo configuration
TIME_RESOLUTION = 30
CHANNELS = [
  'EEG Fpz-Cz',
  'EEG Pz-Oz'
]
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
PK2 = PROBE_KEYS[1]

# -----------------------------------------------------------------------------
# (2) Construct a HypnoStudio
# -----------------------------------------------------------------------------
hs = HypnoStudio(work_dir=STUDIO_DIR, sg_dir=SG_DIR, neb_dir=NEB_DIR)

# -----------------------------------------------------------------------------
# (3) Generate a testing photo
# -----------------------------------------------------------------------------
# print(hs.get_photo_filename('SC4001EC', 'AMP', 'FREQ'))
sg_file_list = finder.walk(SG_DIR, pattern=SG_PATTERN)

sg_path = sg_file_list[3]
neb_fn = os.path.basename(sg_path).split('(')[0]
neb_path = os.path.join(NEB_DIR, neb_fn)

properties = {'Age': 19, 'Gender': 'Male'}

hs.take_one_photo(sg_path, neb_path, CHANNELS, TIME_RESOLUTION, PK1, PK2,
                  show_figure=True, properties=properties, fig_size=(8, 6))


import os.path

from hypnomics.freud import HypnoStudio
from hypnomics.freud.nebula import Nebula
from roma import console, finder, io



# -----------------------------------------------------------------------------
# (1) Configuration
# -----------------------------------------------------------------------------
IN_LINUX = os.name != 'nt'

# (1.1) Working directory & patient inclusion
SG_PATTERN = r'sub-S*_ses-?(float16,128Hz).sg'
if IN_LINUX:
  SG_DIR = r'\\192.168.5.100\xai-beta\xai-sleep\data\hsp\hsp_sg'
  NEB_DIR = r'\\192.168.5.100\xai-beta\xai-sleep\data\hsp\hsp_nebula'
  STUDIO_DIR = r'\\192.168.5.100\xai-beta\xai-sleep\data\hsp\hsp_studio'
else:
  SG_DIR = r'../../data/hsp/hsp_sg'
  NEB_DIR = r'../../data/hsp/hsp_nebula'
  STUDIO_DIR = r'../../data/hsp/hsp_studio'

# (1.2) Photo configuration
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

sg_path = sg_file_list[2]
neb_fn = os.path.basename(sg_path).split('(')[0]
neb_path = os.path.join(NEB_DIR, neb_fn)

properties = {'Age': 70, 'Gender': 'Male'}

hs.take_one_photo(sg_path, neb_path, CHANNELS, TIME_RESOLUTION, PK1, PK2,
                  show_figure=True, properties=properties, fig_size=(9, 8))


# Add path in order to be compatible with Linux
import sys, os

SOLUTION_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
print(f'Solution dir = {SOLUTION_DIR}')

PATH_LIST = ['26-HSP', '26-HSP/01-sg-conversion', 'xai-kit',
             'xai-kit/roma', 'xai-kit/pictor', 'xai-kit/tframe']

sys.path.append(SOLUTION_DIR)
for p in PATH_LIST: sys.path.append(os.path.join(SOLUTION_DIR, p))

# Import anything here
from freud.hypno_tools.probe_tools import get_probe_keys, get_extractor_dict
from hypnomics.freud.freud import Freud
from roma import finder



# -----------------------------------------------------------------------------
# (1) Configuration
# -----------------------------------------------------------------------------
OVERWRITE = 0

SG_PATH = os.path.join(SOLUTION_DIR, 'data/hsp/hsp_sg')
SG_PATTERN = r'sub-S*_ses-?(float16,128Hz).sg'
NEB_PATH = os.path.join(SOLUTION_DIR, 'data/hsp/hsp_nebula')

CHANNELS = ['EEG F3-M2', 'EEG C3-M2', 'EEG O1-M2',
            'EEG F4-M1', 'EEG C4-M1', 'EEG O2-M1']
TIME_RESOLUTION = 30
PROBE_CONFIG = 'A'

PROBE_KEYS = get_probe_keys(PROBE_CONFIG)

PK1 = PROBE_KEYS[0]
PK2 = PROBE_KEYS[1]

N = 999
SG_LABELS = finder.walk(NEB_PATH, type_filter='dir', return_basename=True)[:N]
# -----------------------------------------------------------------------------
# (2) Cloud visualization
# -----------------------------------------------------------------------------
from hypnomics.freud.telescopes.telescope import Telescope

freud = Freud(NEB_PATH)
nebula = freud.load_nebula(sg_labels=SG_LABELS,
                           channels=CHANNELS,
                           time_resolution=TIME_RESOLUTION,
                           probe_keys=PROBE_KEYS)

configs = {
  # 'xmin': 3, 'xmax': 9, 'ymin': 0, 'ymax': 200,
  'show_kde': 1,
  'show_scatter': 0,
  'show_vector': 0,
  # 'scatter_alpha': 0.05,
}
viewer_class = Telescope
nebula.dual_view(x_key=PK1, y_key=PK2, viewer_class=viewer_class,
                 viewer_configs={'plotters': 'HA'}, **configs)

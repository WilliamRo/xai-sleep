from hypnomics.freud.freud import Freud
from hypnomics.freud.nebula import Nebula
from hypnomics.freud.telescopes.telescope import Telescope
from roma import finder

import os



# -----------------------------------------------------------------------------
# (1) Configuration
# -----------------------------------------------------------------------------
configs = {
  # 'xmin': 3, 'xmax': 9, 'ymin': 0, 'ymax': 200,
  'show_kde': 0,
  'show_scatter': 0,
  'show_vector': 1,
  # 'scatter_alpha': 0.05,
}

WORK_DIR = r'../data/mass_alpha'
CHANNELS = [
  ('EEG F3-REF', 'EEG F3-CLE', 'EEG F3-LER'),
  # ('EEG F4-REF', 'EEG F4-CLE', 'EEG F4-LER'),
  ('EEG C3-REF', 'EEG C3-CLE', 'EEG C3-LER'),
  # ('EEG C4-REF', 'EEG C4-CLE', 'EEG C4-LER'),
  ('EEG O1-REF', 'EEG O1-CLE', 'EEG O1-LER'),
  # ('EEG O2-REF', 'EEG O2-CLE', 'EEG O2-LER'),
]

PROBE_KEYS = [
  'FREQ-20',   # 0
  # 'GFREQ-35',  # 1
  'AMP-1',     # 2
  # 'P-TOTAL',   # 3
  # 'RP-DELTA',  # 4
  # 'RP-THETA',  # 5
  # 'RP-ALPHA',  # 6
  # 'RP-BETA',   # 7
]
PK1 = PROBE_KEYS[0]
PK2 = PROBE_KEYS[1]

# SG_LABELS = ['SC4001E', 'SC4002E']
N = 10
SG_LABELS = finder.walk(WORK_DIR, type_filter='dir', return_basename=True)[:N]

# [ 2, 5, 10, 30, ]
TIME_RESOLUTION = 10

NEB_FN = [
  'None',
][0]
# -----------------------------------------------------------------------------
# (2) Visualize
# -----------------------------------------------------------------------------
neb_file_path = os.path.join(WORK_DIR, NEB_FN)
if NEB_FN != 'None' and os.path.exists(neb_file_path):
  nebula: Nebula = Nebula.load(neb_file_path)
else:
  freud = Freud(WORK_DIR)
  nebula = freud.load_nebula(sg_labels=SG_LABELS,
                             channels=CHANNELS,
                             time_resolution=TIME_RESOLUTION,
                             probe_keys=PROBE_KEYS)

viewer_class = Telescope
nebula.dual_view(x_key=PK1, y_key=PK2, viewer_class=viewer_class,
                 viewer_configs={'plotters': 'HA'}, **configs)

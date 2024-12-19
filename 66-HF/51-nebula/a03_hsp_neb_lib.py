from freud.hypno_tools.probe_tools import get_probe_keys
from hypnomics.freud.freud import Freud
from hypnomics.freud.nebula import Nebula
from hypnomics.freud.telescopes.telescope import Telescope
from hypnomics.freud.telescopes.popglass import PopGlass
from roma import console, finder

import os



# -----------------------------------------------------------------------------
# (1) Configuration
# -----------------------------------------------------------------------------
IN_LINUX = os.name != 'nt'

# (1.1) Configure directories
if IN_LINUX:
  NEB_DIR = r'\\192.168.5.100\xai-beta\xai-sleep\data\hsp\hsp_nebula'
  POP_DIR = r'\\192.168.5.100\xai-beta\xai-sleep\data\hsp\hsp_population'
else:
  NEB_DIR = r'../../data/hsp/hsp_nebula'
  POP_DIR = r'../../data/hsp/hsp_population'

# (1.2) Fixed configuration, do not modify
TIME_RESOLUTION = 30

CHANNELS = ['EEG F3-M2', 'EEG C3-M2', 'EEG O1-M2',
            'EEG F4-M1', 'EEG C4-M1', 'EEG O2-M1']

# (1.3) TODO: configure here
# Ab: 'POWER-30', 'PR-DELTA_TOTAL', 'PR-THETA_TOTAL', 'PR-ALPHA_TOTAL',
#     'PR-BETA_TOTAL', 'PR-SIGMA_TOTAL', 'PR-DELTA_THETA', 'PR-DELTA_ALPHA',
#     'PR-THETA_ALPHA', 'FREQ-20', 'AMP-1'
PROBE_CONFIG = 'A'
PROBE_KEYS = get_probe_keys(PROBE_CONFIG, expand_group=True)

OVERWRITE = 0

# (1.*) Auto configuration
NEB_FN = f'HSP-000-{PROBE_CONFIG}-{len(CHANNELS)}chn-{TIME_RESOLUTION}s.nebula'
NEB_PATH = os.path.join(POP_DIR, NEB_FN)
# -----------------------------------------------------------------------------
# (2) Manage nebula library
# -----------------------------------------------------------------------------
if os.path.exists(NEB_PATH) and not OVERWRITE:
  nebula: Nebula = Nebula.load(NEB_PATH, verbose=True)
else:
  freud = Freud(NEB_DIR)
  SG_LABELS = finder.walk(NEB_DIR, type_filter='dir', return_basename=True)
  nebula = freud.load_nebula(sg_labels=SG_LABELS,
                             channels=CHANNELS,
                             time_resolution=TIME_RESOLUTION,
                             probe_keys=PROBE_KEYS)
  nebula.save(NEB_PATH, verbose=True)

# TODO: remove bad clouds
nebula.labels.remove('sub-S0001111190905_ses-4')
# -----------------------------------------------------------------------------
# (3) Visualize nebula
# -----------------------------------------------------------------------------
PK1 = 'FREQ-20'
PK2 = 'AMP-1'

viewer_class = [Telescope, PopGlass][1]
if viewer_class is Telescope:
  configs = {
    'show_kde': 0,
    'show_scatter': 1,
    'show_vector': 0,
  }
  nebula.dual_view(x_key=PK1, y_key=PK2, viewer_class=viewer_class,
                   viewer_configs={'plotters': 'HA'}, **configs)
else:
  nebula.dual_view(x_key=PK1, y_key=PK2, viewer_class=viewer_class)

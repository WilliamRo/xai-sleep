from freud.hypno_tools.probe_tools import get_probe_keys
from hf.sc_tools import load_nebula_from_clouds
from hypnomics.freud.freud import Freud
from hypnomics.freud.nebula import Nebula
from hypnomics.freud.telescopes.telescope import Telescope
from hypnomics.freud.telescopes.popglass import PopGlass
from roma import console, finder
from sc.sc_agent import SCAgent

import os



# -----------------------------------------------------------------------------
# (1) Configuration
# -----------------------------------------------------------------------------
# (1.1) Configure directories
NEB_DIR = r'../data/sleepedfx_sc'
POP_DIR = r'../../data/sleepedfx-sc/sc_population'

XLSX_PATH = r'../../data/sleep-edf-database-expanded-1.0.0/SC-subjects.xls'

# (1.2) Fixed configuration, do not modify
TIME_RESOLUTION = 30

CHANNELS = [
  'EEG Fpz-Cz',
  'EEG Pz-Oz'
]

# (1.3) TODO: configure here
# Ab: 'POWER-30', 'PR-DELTA_TOTAL', 'PR-THETA_TOTAL', 'PR-ALPHA_TOTAL',
#     'PR-BETA_TOTAL', 'PR-SIGMA_TOTAL', 'PR-DELTA_THETA', 'PR-DELTA_ALPHA',
#     'PR-THETA_ALPHA', 'FREQ-20', 'AMP-1'
PROBE_CONFIG = 'Ab'
PROBE_KEYS = get_probe_keys(PROBE_CONFIG, expand_group=True)

OVERWRITE = 0

# (1.*) Auto configuration
NEB_FN = f'SC-153-{PROBE_CONFIG}-{len(CHANNELS)}chn-{TIME_RESOLUTION}s.nebula'
# NEB_FN = 'SC-153-Ab-2chn-30s.nebula'   # 1
NEB_FN = 'SC-153-Ab-2chn-30s-buffer_FA.nebula'   # 2
NEB_PATH = os.path.join(POP_DIR, NEB_FN)
# -----------------------------------------------------------------------------
# (2) Manage nebula library
# -----------------------------------------------------------------------------
if os.path.exists(NEB_PATH) and not OVERWRITE:
  nebula: Nebula = Nebula.load(NEB_PATH, verbose=True)
else:
  SG_LABELS = finder.walk(NEB_DIR, type_filter='dir', return_basename=True)
  nebula = load_nebula_from_clouds(NEB_DIR, SG_LABELS, CHANNELS,
                                   TIME_RESOLUTION, PROBE_KEYS, XLSX_PATH)

  nebula.save(NEB_PATH, verbose=True)

# -----------------------------------------------------------------------------
# (3) Visualize nebula
# -----------------------------------------------------------------------------
PK1, PK2 = 'FREQ-20', 'AMP-1'
# PK1, PK2 = 'PR-ALPHA_TOTAL', 'PR-DELTA_TOTAL'
# PK1, PK2 = 'FREQ-20', 'PR-THETA_TOTAL'
# PK1, PK2 = 'FREQ-20', 'PR-ALPHA_TOTAL'

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

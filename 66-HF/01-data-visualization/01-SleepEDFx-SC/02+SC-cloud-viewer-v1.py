# Add path in order to be compatible with Linux
import sys, os

SOLUTION_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))

PATH_LIST = ['32-SC', '66-HF', 'xai-kit', 'hypnomics',
             'xai-kit/roma', 'xai-kit/pictor', 'xai-kit/tframe']

if __name__ == '__main__':
  print(f'Solution dir = {SOLUTION_DIR}')
  sys.path.append(SOLUTION_DIR)
  for p in PATH_LIST: sys.path.append(os.path.join(SOLUTION_DIR, p))

# -----------------------------------------------------------------------------
from freud.hypno_tools.probe_tools import get_probe_keys
from hypnomics.freud.freud import Freud
from hypnomics.freud.nebula import Nebula
from roma import finder, console

import os
import sc as hub



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

CHANNELS = [
  'EEG Fpz-Cz',
  'EEG Pz-Oz'
]

PROBE_CONFIG = 'Ab'
PROBE_KEYS = get_probe_keys(PROBE_CONFIG, expand_group=True)

console.show_status(f'Number of digital probes: {len(PROBE_KEYS)}')

# SG_LABELS = ['SC4001E', 'SC4002E']
N = 999
SG_LABELS = finder.walk(hub.CLOUD_DIR, type_filter='dir', return_basename=True)[:N]

# [ 2, 5, 10, 30, ]
TIME_RESOLUTION = 30
if len(sys.argv) == 2: TIME_RESOLUTION = int(sys.argv[1])
assert TIME_RESOLUTION in [2, 5, 10, 30]

NEB_FN = [
  'None',
  f'SC-153-partial-{TIME_RESOLUTION}.nebula',
  f'SC-{TIME_RESOLUTION}-KDE-0730.nebula',
  f'SC-{TIME_RESOLUTION}-Ad11.nebula',
  hub.get_neb_file_name(TIME_RESOLUTION, PROBE_CONFIG),
][-1]

PK1 = 'PR-ALPHA_TOTAL'
PK2 = 'PR-DELTA_TOTAL'
# -----------------------------------------------------------------------------
# (2) Visualize
# -----------------------------------------------------------------------------
neb_file_path = os.path.join(hub.NEBULA_DIR, NEB_FN)
if NEB_FN != 'None' and os.path.exists(neb_file_path):
  nebula: Nebula = Nebula.load(neb_file_path)
else:
  freud = Freud(hub.CLOUD_DIR)
  nebula = freud.load_nebula(sg_labels=SG_LABELS,
                             channels=CHANNELS,
                             time_resolution=TIME_RESOLUTION,
                             probe_keys=PROBE_KEYS)
  nebula.save(neb_file_path)

if not hub.IN_LINUX:
  from hypnomics.freud.telescopes.telescope import Telescope

  viewer_class = Telescope
  nebula.dual_view(x_key=PK1, y_key=PK2, viewer_class=viewer_class,
                   viewer_configs={'plotters': 'HA'}, **configs)

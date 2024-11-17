from hf.probe_tools import get_probe_keys
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
}

WORK_DIR = r'../data/sleepedfx_sc'
CHANNELS = [
  'EEG Fpz-Cz',
  'EEG Pz-Oz'
]

PROBE_CONFIG = 'ABD'
PROBE_KEYS = get_probe_keys(PROBE_CONFIG)

PK1 = PROBE_KEYS[0]
PK2 = PROBE_KEYS[1]

# SG_LABELS = ['SC4001E', 'SC4002E']
N = 999
SG_LABELS = finder.walk(WORK_DIR, type_filter='dir', return_basename=True)[:N]

# [ 2, 5, 10, 30, ]
TIME_RESOLUTION = 30

NEB_FN = [
  'None',
  f'SC-{TIME_RESOLUTION}s-KDE-39-probes.nebula',
][1]
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
  nebula.save(neb_file_path)

viewer_class = Telescope
nebula.set_probe_keys(PROBE_KEYS)
nebula.dual_view(x_key=PK1, y_key=PK2, viewer_class=viewer_class,
                 viewer_configs={'plotters': 'HA'}, **configs)

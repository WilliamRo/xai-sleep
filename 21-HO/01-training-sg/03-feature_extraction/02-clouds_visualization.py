import numpy as np

from hypnomics.freud.freud import Freud
from hypnomics.freud.telescopes.telescope import Telescope
from roma import finder




# -----------------------------------------------------------------------------
# (1) Configuration
# -----------------------------------------------------------------------------
# Specify the directory containing clouds files
WORK_DIR = r'../data/sleepedfx_sc'

# [ 2, 5, 10, 30, ]
TIME_RESOLUTION = 10

# Specify channels to visualize
CHANNELS = [
  'EEG Fpz-Cz',
  'EEG Pz-Oz'
]

# Specify probe keys (corresponding to X and Y axis)
PK1 = 'FREQ-20'
PK2 = 'AMP-1'

# SG_LABELS = ['SC4001E', 'SC4002E']
N = 6
SG_LABELS = finder.walk(WORK_DIR, type_filter='dir', return_basename=True)[:N]

# Configure viewer
configs = {
  'show_kde': 1,
  'show_scatter': 0,
  'show_vector': 0,
}
# -----------------------------------------------------------------------------
# (2) Visualize
# -----------------------------------------------------------------------------
freud = Freud(WORK_DIR)
nebula = freud.load_nebula(sg_labels=SG_LABELS,
                           channels=CHANNELS,
                           time_resolution=TIME_RESOLUTION,
                           probe_keys=[PK1, PK2])

# Put random age and gender into nebula.meta for demo only
for pid in nebula.labels:
  nebula.meta[pid] = {'age': np.random.randint(20, 100),
                      'gender': np.random.choice(['M', 'F'])}

viewer_configs = {'plotters': 'HA', 'meta_keys': ('age', 'gender')}
nebula.dual_view(x_key=PK1, y_key=PK2, viewer_configs=viewer_configs,
                 viewer_class=Telescope, **configs)


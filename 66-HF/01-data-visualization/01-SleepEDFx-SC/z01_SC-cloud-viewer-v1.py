import os

import numpy as np

from collections import OrderedDict
from hypnomics.freud.freud import Freud
from hypnomics.freud.nebula import Nebula
from hypnomics.freud.telescopes.telescope import Telescope
from roma import finder



# -----------------------------------------------------------------------------
# (1) Configuration
# -----------------------------------------------------------------------------
# sca = SCAgent()
# sca.report_data_info()
configs = {
  # 'xmin': 3, 'xmax': 9, 'ymin': 0, 'ymax': 200,
  'show_kde': 0,
  'show_scatter': 0,
  'show_vector': 1,
  # 'scatter_alpha': 0.05,
}

WORK_DIR = r'../../data/sleepedfx_sc'
CHANNELS = [
  'EEG Fpz-Cz',
  'EEG Pz-Oz'
]

PROBE_KEYS = [
  'P-TOTAL',   # 1
  'POWER-30',   # 2
  'RP-DELTA',  # 3
  'PR-DELTA_TOTAL',  # 4
  'RP-THETA',  # 5
  'PR-THETA_TOTAL',  # 6
  'RP-ALPHA',  # 7
  'PR-ALPHA_TOTAL',  # 8
]
PK1 = PROBE_KEYS[3]
PK2 = PROBE_KEYS[7]

# SG_LABELS = ['SC4001E', 'SC4002E']
N = 999
SG_LABELS = finder.walk(WORK_DIR, type_filter='dir', return_basename=True)[:N]

# [ 2, 5, 10, 30, ]
TIME_RESOLUTION = 30

# -----------------------------------------------------------------------------
# (2) Visualize
# -----------------------------------------------------------------------------
freud = Freud(WORK_DIR)
nebula = freud.load_nebula(sg_labels=SG_LABELS,
                           channels=CHANNELS,
                           time_resolution=TIME_RESOLUTION,
                           probe_keys=PROBE_KEYS)

# viewer_class = Telescope
# nebula.dual_view(x_key=PK1, y_key=PK2, viewer_class=viewer_class,
#                  viewer_configs={'plotters': 'HA'}, **configs)

# -----------------------------------------------------------------------------
# (3) Calculate deltas
# -----------------------------------------------------------------------------
STAGE_KEYS = ['W', 'N1', 'N2', 'N3', 'R']

key_pairs = [('P-TOTAL', 'POWER-30'), ('RP-DELTA', 'PR-DELTA_TOTAL'),
             ('RP-THETA', 'PR-THETA_TOTAL'), ('RP-ALPHA', 'PR-ALPHA_TOTAL')]

delta_dict = OrderedDict()
for key_mt, key_wel in key_pairs:
  mt_list, wel_list = [], []
  for sg_label in nebula.labels:
    for ck in CHANNELS:
      mt_dict = nebula.data_dict[(sg_label, ck, key_mt)]
      wel_dict = nebula.data_dict[(sg_label, ck, key_wel)]

      mt_list.append(np.concatenate([mt_dict[sk] for sk in STAGE_KEYS]))
      wel_list.append(np.concatenate([wel_dict[sk] for sk in STAGE_KEYS]))

  mt_values = np.concatenate(mt_list)
  wel_values = np.concatenate(wel_list)
  delta_dict[(key_mt, key_wel)] = (mt_values, wel_values)

# -----------------------------------------------------------------------------
# (4) Draw distribution
# -----------------------------------------------------------------------------
from pictor import Pictor
import matplotlib.pyplot as plt

p = Pictor()
def plotter(x, ax: plt.Axes):
  keys, values = x
  mt, wel = values
  ax.hist(mt, bins=100, alpha=0.5, label=f'{keys[0]}')
  ax.hist(wel, bins=100, alpha=0.5, label=f'{keys[1]}')
  ax.hist(mt-wel, bins=100, alpha=0.5, label=f'{keys[0]} - {keys[1]}')
  ax.legend()

p.add_plotter(plotter)
p.objects = [(k, v) for k, v in delta_dict.items()]
p.show()


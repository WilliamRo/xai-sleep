from hypnomics.freud.freud import Freud
from hypnomics.freud.telescopes.telescope import Telescope
from roma import finder

import pandas as pd
import numpy as np



# -----------------------------------------------------------------------------
# (1) Configuration
# -----------------------------------------------------------------------------
# (1.1) Nebula configuration
WORK_DIR = r'../data/sleepedfx_sc'
SG_LABELS = finder.walk(WORK_DIR, type_filter='dir', return_basename=True)

# [ 2(x), 5(x), 10(x), 30, ]
TIME_RESOLUTION = 30

CHANNELS = [
  'EEG Fpz-Cz',
  'EEG Pz-Oz'
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

# (1.2) Dual view configuration
PK1 = PROBE_KEYS[0]
PK2 = PROBE_KEYS[1]

# (1.3) Excel path
XLSX_PATH = r'../../data/sleep-edf-database-expanded-1.0.0/SC-subjects.xls'
# -----------------------------------------------------------------------------
# (2) Load nebula
# -----------------------------------------------------------------------------
# (2.1) Load nebula
freud = Freud(WORK_DIR)
nebula = freud.load_nebula(sg_labels=SG_LABELS,
                           channels=CHANNELS,
                           time_resolution=TIME_RESOLUTION,
                           probe_keys=PROBE_KEYS)

# (2.2) Set basic infor
df = pd.read_excel(XLSX_PATH)
# -----------------------------------------------------------------------------
# (3) Visualization
# -----------------------------------------------------------------------------
configs = {
  # 'xmin': 3, 'xmax': 9, 'ymin': 0, 'ymax': 200,
  'show_kde': 0,
  'show_scatter': 1,
  'show_vector': 0,
  # 'scatter_alpha': 0.05,
}

viewer_class = Telescope
nebula.dual_view(x_key=PK1, y_key=PK2, viewer_class=viewer_class,
                 viewer_configs={'plotters': 'HA'}, **configs)

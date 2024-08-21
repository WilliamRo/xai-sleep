from hypnomics.hypnoprints.extractor import Extractor
from hypnomics.freud.freud import Freud
from roma import finder
from pictor.xomics.omix import Omix

import pandas as pd
import numpy as np



# -----------------------------------------------------------------------------
# (1) Configuration
# -----------------------------------------------------------------------------
# (1.1) Target nebula file path
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
  'P-TOTAL',   # 3
  'RP-DELTA',  # 4
  'RP-THETA',  # 5
  'RP-ALPHA',  # 6
  'RP-BETA',   # 7
]

XLSX_PATH = r'../../data/sleep-edf-database-expanded-1.0.0/SC-subjects.xls'
# -----------------------------------------------------------------------------
# (2) Load paired nebula
# -----------------------------------------------------------------------------
freud = Freud(WORK_DIR)
nebula = freud.load_nebula(sg_labels=SG_LABELS,
                           channels=CHANNELS,
                           time_resolution=TIME_RESOLUTION,
                           probe_keys=PROBE_KEYS)

df = pd.read_excel(XLSX_PATH)
# -----------------------------------------------------------------------------
# (3) Omix construction
# -----------------------------------------------------------------------------
extractor = Extractor()
feature_dict = extractor.extract(nebula, return_dict=True)
features = np.stack([np.array(list(v.values()))
                     for v in feature_dict.values()], axis=0)
feature_names = list(list(feature_dict.values())[0].keys())

targets = [
  df.loc[df['subject'] == int(pid[3:5]), 'age'].values[0]
  for pid in nebula.labels
]

# target_labels = ['Age']

T = 60
targets = [int(t > T) for t in targets]
target_labels = [f'Age<={T}', f'Age>{T}']

omix = Omix(features, targets, feature_names, None, target_labels,
            data_name=f'SC-153x375-{TIME_RESOLUTION}s')
omix.show_in_explorer()

from mass.mass_tools import load_nebula_from_clouds
from hypnomics.hypnoprints.extractor import Extractor
from pictor.xomics.omix import Omix
from roma import finder

import numpy as np



# -----------------------------------------------------------------------------
# (1) Configuration
# -----------------------------------------------------------------------------
# (1.1) Nebula configuration
WORK_DIR = r'../data/mass_alpha'   # contains cloud files
SG_LABELS = finder.walk(WORK_DIR, type_filter='dir', return_basename=True,
                        pattern='01*')

# [ 2(x), 5(x), 10, 30(x) ]
TIME_RESOLUTION = 10

CHANNELS = [
  ('EEG F3-REF', 'EEG F3-CLE', 'EEG F3-LER'),
  ('EEG F4-REF', 'EEG F4-CLE', 'EEG F4-LER'),
  ('EEG C3-REF', 'EEG C3-CLE', 'EEG C3-LER'),
  ('EEG C4-REF', 'EEG C4-CLE', 'EEG C4-LER'),
  ('EEG O1-REF', 'EEG O1-CLE', 'EEG O1-LER'),
  ('EEG O2-REF', 'EEG O2-CLE', 'EEG O2-LER'),
]

PROBE_KEYS = [
  'FREQ-20',   # 0
  # 'GFREQ-35',  # 1
  'AMP-1',     # 2
  # 'P-TOTAL',   # 3
  'RP-DELTA',  # 4
  'RP-THETA',  # 5
  'RP-ALPHA',  # 6
  'RP-BETA',   # 7
]

# (1.2) Excel path
XLSX_PATH = r"D:\data\01-MASS\Open-access descriptors\Open-access descriptors v2.xlsx"

# (1.3) Target key
TARGET = 'reference'
# -----------------------------------------------------------------------------
# (2) Load nebula, generate evolution
# -----------------------------------------------------------------------------
nebula = load_nebula_from_clouds(WORK_DIR, SG_LABELS, CHANNELS, TIME_RESOLUTION,
                                 PROBE_KEYS, XLSX_PATH)



if __name__ == '__main__':
  extractor = Extractor()
  feature_dict = extractor.extract(nebula, return_dict=True)
  features = np.stack([np.array(list(v.values()))
                       for v in feature_dict.values()], axis=0)
  feature_names = list(list(feature_dict.values())[0].keys())

  target_labels = ['CLE', 'LER']
  targets = [nebula.meta[pid][TARGET] for pid in nebula.labels]
  omix = Omix(features, targets, feature_names, nebula.labels, target_labels,
              data_name=f'MASS-{TARGET}-53-{TIME_RESOLUTION}s')

  omix.show_in_explorer()

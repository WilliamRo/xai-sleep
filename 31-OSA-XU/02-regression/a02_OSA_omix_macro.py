from osaxu.osa_tools import load_macro_and_meta_from_workdir
from hypnomics.hypnoprints.extractor import Extractor
from pictor.xomics.omix import Omix
from roma import finder

import numpy as np
import os



# -----------------------------------------------------------------------------
# (1) Configuration
# -----------------------------------------------------------------------------
# (1.1) Path configuration
WORK_DIR = r'../data'  # contains cloud files
SG_LABELS = finder.walk(WORK_DIR, type_filter='dir', return_basename=True)

# (1.2) Excel path
XLSX_PATH = r"P:\xai-sleep\data\rrsh-osa\OSA-xu.xlsx"

# (1.3) Target
TARGET = [
  'AHI',
  'age'
][0]

# (1.4) MISC
INCLUDE_TRANSITION_PER_HOUR = 0
LOG = 1
# -----------------------------------------------------------------------------
# (2) Load macro features and meta data
# -----------------------------------------------------------------------------
x_dict, meta_dict = load_macro_and_meta_from_workdir(
  WORK_DIR, SG_LABELS, XLSX_PATH)

features = np.stack([list(x_dict[pid].values()) for pid in SG_LABELS], axis=0)

if LOG:
  # Notice that features > 0, thus sign(x)log(|x| + 1) is not necessary
  features = np.log(features + 1)

feature_names = list(x_dict[SG_LABELS[0]].keys())

if not INCLUDE_TRANSITION_PER_HOUR:
  features = features[:, :-1]
  feature_names = feature_names[:-1]

n_features = 30 + INCLUDE_TRANSITION_PER_HOUR
assert features.shape[1] == len(feature_names) == n_features
# -----------------------------------------------------------------------------
# (3) Wrap data into Omix
# -----------------------------------------------------------------------------
target_labels = [TARGET]
targets = [meta_dict[pid][TARGET] for pid in SG_LABELS]

data_name = f'OSA-{TARGET}-macro-{n_features}'
omix = Omix(features, targets, feature_names, SG_LABELS, target_labels,
            data_name=data_name)



"""
TARGET = AHI:
  - simple 'ml eln' yields approx MAE = 8.8 when LOG=1
"""

if __name__ == '__main__':
  omix.show_in_explorer()


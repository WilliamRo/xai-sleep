from hf.sc_tools import load_macro_and_meta_from_workdir
from pictor.xomics.omix import Omix
from roma import finder

import numpy as np



# -----------------------------------------------------------------------------
# (1) Configuration
# -----------------------------------------------------------------------------
WORK_DIR = r'../data/sleepedfx_sc'
SG_LABELS = finder.walk(WORK_DIR, type_filter='dir', return_basename=True)

FEATURE_PATS = ['*_Percentage', 'Transition_Probability_*_to_*']
INCLUDE_TRANSITION_PER_HOUR = 0
LOG = 1

# (1.2) Excel path
XLSX_PATH = r'../../data/sleep-edf-database-expanded-1.0.0/SC-subjects.xls'

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
target_labels = ['Age']
targets = [meta_dict[pid]['age'] for pid in SG_LABELS]

data_name = f'SC-age-macro-{n_features}'
if LOG: data_name += '-log'
omix = Omix(features, targets, feature_names, SG_LABELS, target_labels,
            data_name=data_name)

omix.show_in_explorer()

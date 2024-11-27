from hypnomics.freud.nebula import Nebula
from osaxu.osa_tools import load_macro_and_meta_from_workdir
from osaxu.osa_tools import set_target_collection_for_omix
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
][1]

# (1.4) MISC
INCLUDE_TRANSITION_PER_HOUR = 0

OMIX_FN = f'OSA_macro_{TARGET}.omix'
OMIX_PATH = os.path.join(WORK_DIR, OMIX_FN)

NEB_FN = f'125samples-6channels-39probes-30s.nebula'
NEB_PATH = os.path.join(WORK_DIR, NEB_FN)

OVERWRITE = 0
# -----------------------------------------------------------------------------
# (2) Load macro features and meta data
# -----------------------------------------------------------------------------
if os.path.exists(OMIX_PATH) and not OVERWRITE:
  omix = Omix.load(OMIX_PATH)
else:
  x_dict, meta_dict = load_macro_and_meta_from_workdir(
    WORK_DIR, SG_LABELS, XLSX_PATH)

  features = np.stack([list(x_dict[pid].values()) for pid in SG_LABELS], axis=0)

  feature_names = list(x_dict[SG_LABELS[0]].keys())

  if not INCLUDE_TRANSITION_PER_HOUR:
    features = features[:, :-1]
    feature_names = feature_names[:-1]

  n_features = 30 + INCLUDE_TRANSITION_PER_HOUR
  assert features.shape[1] == len(feature_names) == n_features
  target_labels = [TARGET]
  targets = [meta_dict[pid][TARGET] for pid in SG_LABELS]

  data_name = f'OSA-{TARGET}-macro-{n_features}'
  omix = Omix(features, targets, feature_names, SG_LABELS, target_labels,
              data_name=data_name)

  nebula = Nebula.load(NEB_PATH)
  set_target_collection_for_omix(omix, nebula)

  omix.save(OMIX_PATH)



if __name__ == '__main__':
  omix.show_in_explorer()


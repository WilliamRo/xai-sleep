"""
Last modified: 2024-12-25

This script is for generating macro omix given nebula.
Ref: 66-HF/03-sleep-age/m01_macro_omix_gen.py
"""
# Add path in order to be compatible with Linux
import sys, os

SOLUTION_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

PATH_LIST = ['26-HSP', '26-HSP/99-data-probe', 'xai-kit', 'hypnomics',
             'xai-kit/roma', 'xai-kit/pictor', 'xai-kit/tframe']

if __name__ == '__main__':
  print(f'Solution dir = {SOLUTION_DIR}')
  sys.path.append(SOLUTION_DIR)
  for p in PATH_LIST: sys.path.append(os.path.join(SOLUTION_DIR, p))

from hypnomics.freud.nebula import Nebula
from pictor.xomics.omix import Omix
from roma import io

import a00_common as hub
import numpy as np
import re



# -----------------------------------------------------------------------------
# (0) Configuration
# -----------------------------------------------------------------------------
OVERWRITE = 0

INCLUDE_TRANSITION_PER_HOUR = False

NEB_FN = f'HSP-100-E-6chn-30s.nebula'
OMIX_FN = re.search(r'HSP-\d+', NEB_FN).group(0) + '-macro.omix'
OMIX_PATH = os.path.join(hub.OMIX_DIR, OMIX_FN)

# -----------------------------------------------------------------------------
# (1) Load or crate .omix
# -----------------------------------------------------------------------------
if os.path.exists(OMIX_PATH) and not OVERWRITE:
  omix: Omix = Omix.load(OMIX_PATH, verbose=True)
else:
  # (1.1) Load nebula
  NEB_PATH = os.path.join(hub.OMIX_DIR, NEB_FN)
  nebula: Nebula = Nebula.load(NEB_PATH, verbose=True)
  SG_LABELS = nebula.labels

  # (1.2) Load features
  x_dict = {}
  for pid in SG_LABELS:
    macro_path = os.path.join(hub.CLOUD_DIR, pid, 'macro_alpha.od')
    x_dict[pid] = io.load_file(macro_path)

  # (1.3) Prepare features
  features = np.stack([list(x_dict[pid].values()) for pid in SG_LABELS], axis=0)
  feature_names = list(x_dict[SG_LABELS[0]].keys())

  if not INCLUDE_TRANSITION_PER_HOUR:
    features = features[:, :-1]
    feature_names = feature_names[:-1]

  # (1.3.1) Sanity check
  n_features = 30 + INCLUDE_TRANSITION_PER_HOUR
  assert features.shape[1] == len(feature_names) == n_features

  # (1.4) Wrap data into .omix
  target_labels = ['Age']
  targets = [nebula.meta[pid]['age'] for pid in SG_LABELS]

  data_name = f'HSP-age-macro-{n_features}'
  omix = Omix(features, targets, feature_names, SG_LABELS, target_labels,
              data_name=data_name)

  # (1.5) Save data
  omix.save(OMIX_PATH, verbose=True)

# -----------------------------------------------------------------------------
# (2) Show in explorer
# -----------------------------------------------------------------------------
if not hub.IN_LINUX: omix.show_in_explorer()

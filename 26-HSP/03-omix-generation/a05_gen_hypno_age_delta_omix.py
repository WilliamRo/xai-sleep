"""
Last modified: 2024-12-25

This script is for generating omix given nebula
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
from hypnomics.hypnoprints.extractor import Extractor
from pictor.xomics.omix import Omix
from roma import io

import a00_common as hub
import numpy as np



# -----------------------------------------------------------------------------
# (0) Configuration
# -----------------------------------------------------------------------------
SUBSET_FN = hub.SubsetDicts.ss_2ses_3types_378

OVERWRITE = 0

NEB_FN = f'HSP-378-Ab-6chn-30s.nebula'
OMIX_FN = NEB_FN.replace('.nebula', '.omix')

MIN_AD = 0
# -----------------------------------------------------------------------------
# (1) Load omix and patient dict
# -----------------------------------------------------------------------------
OMIX_PATH = os.path.join(hub.OMIX_DIR, OMIX_FN)
assert os.path.exists(OMIX_PATH)
omix: Omix = Omix.load(OMIX_PATH, verbose=True)

subset_dict_path = os.path.join(SOLUTION_DIR, 'data/hsp', SUBSET_FN)
assert os.path.exists(subset_dict_path)
patient_dict: dict = io.load_file(subset_dict_path, verbose=True)

# -----------------------------------------------------------------------------
# (2) Generate omix
# -----------------------------------------------------------------------------
label_pairs, ad_dict = hub.ha.get_longitudinal_pairs(patient_dict, True)

features, targets, sample_labels = [], [], []
for label_pair in label_pairs:
  ad = ad_dict[label_pair]
  if ad < MIN_AD: continue

  lb1, lb2 = label_pair
  targets.append(ad)
  sample_labels.append(f'{lb1} -> {lb2}')

  i1 = omix.sample_labels.index(lb1)
  i2 = omix.sample_labels.index(lb2)
  feature = omix.features[i2] - omix.features[i1]
  features.append(feature)

feature_names = [f'Diff({name})' for name in omix.feature_labels]
target_labels = ['Delta Age']

features = np.stack(features, axis=0)
omix = Omix(features, targets, feature_names, sample_labels, target_labels,
            data_name=omix.data_name)

omix.show_in_explorer()

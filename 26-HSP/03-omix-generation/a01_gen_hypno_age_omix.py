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

import a00_common as hub
import numpy as np



# -----------------------------------------------------------------------------
# (0) Configuration
# -----------------------------------------------------------------------------
OVERWRITE = 0

NEB_FN = f'HSP-100-E-6chn-30s.nebula'
OMIX_FN = NEB_FN.replace('.nebula', '.omix')
# -----------------------------------------------------------------------------
# (1) Load nebula
# -----------------------------------------------------------------------------
NEB_PATH = os.path.join(hub.OMIX_DIR, NEB_FN)
nebula: Nebula = Nebula.load(NEB_PATH, verbose=True)

# -----------------------------------------------------------------------------
# (2) Generate omix
# -----------------------------------------------------------------------------
OMIX_PATH = os.path.join(hub.OMIX_DIR, OMIX_FN)

if os.path.exists(OMIX_PATH) and not OVERWRITE:
  omix: Omix = Omix.load(OMIX_PATH, verbose=True)
else:
  extractor = Extractor()
  feature_dict = extractor.extract(nebula, return_dict=True)
  features = np.stack([np.array(list(v.values()))
                       for v in feature_dict.values()], axis=0)
  feature_names = list(list(feature_dict.values())[0].keys())

  target_labels = ['Age']
  targets = [nebula.meta[pid]['age'] for pid in nebula.labels]
  omix = Omix(features, targets, feature_names, nebula.labels, target_labels,
              data_name=f'{OMIX_FN.split(".")[0]}')

  omix.save(OMIX_PATH, verbose=True)

omix.show_in_explorer()

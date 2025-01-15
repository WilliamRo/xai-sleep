"""
Last modified: 2024-12-25

This script is for generating macro features (D=31) for existing .sg files
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

from hypnomics.freud.freud import Freud

import a00_common as hub



# -----------------------------------------------------------------------------
# (1) Configuration
# -----------------------------------------------------------------------------
OVERWRITE = 0
SG_PATTERN = f'sub-*(float16,128Hz).sg'

sg_file_list = None
# -----------------------------------------------------------------------------
# (2) Extract macro features
# -----------------------------------------------------------------------------
freud = Freud(hub.CLOUD_DIR)

freud.generate_macro_features(hub.SG_DIR, pattern=SG_PATTERN, config='alpha',
                              overwrite=OVERWRITE, sg_file_list=sg_file_list)

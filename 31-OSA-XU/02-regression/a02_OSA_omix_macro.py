from pictor.xomics.omix import Omix

import os



# -----------------------------------------------------------------------------
# (1) Configuration
# -----------------------------------------------------------------------------
# (1.1) Nebula configuration
WORK_DIR = r'../data'  # contains cloud files

# (1.2) Set path
OMIX_FN = 'OSA_macro.omix'
OMIX_PATH = os.path.join(WORK_DIR, OMIX_FN)

OVERWRITE = 0
TARGET = [
  'AHI',      # 0
  'age',      # 1
  'gender',   # 2
  'MMSE',     # 3
  'cog_imp',  # 4
  'dep',      # 5
  'anx',      # 6
  'som',      # 7
][1]
# -----------------------------------------------------------------------------
# (2) Load omix
# -----------------------------------------------------------------------------
assert os.path.exists(OMIX_PATH)

omix = Omix.load(OMIX_PATH)
omix = omix.set_targets(TARGET, return_new_omix=True)



if __name__ == '__main__':
  omix.show_in_explorer()


"""
TARGET = AHI:
  - simple 'ml eln' yields approx MAE = 8.8 when LOG=1
"""



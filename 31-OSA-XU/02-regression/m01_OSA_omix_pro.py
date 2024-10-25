from osaxu.osa_tools import load_nebula_from_clouds
from hypnomics.hypnoprints.extractor import Extractor
from pictor.xomics.omix import Omix
from roma import finder

import numpy as np
import os



# -----------------------------------------------------------------------------
# (1) Configuration
# -----------------------------------------------------------------------------
# (1.1) Nebula configuration
WORK_DIR = r'../data'  # contains cloud files

# (1.2) Set path
NEB_FN = f'125samples-6channels-39probes-30s.nebula'
NEB_PATH = os.path.join(WORK_DIR, NEB_FN)
OMIX_PATH = NEB_PATH.replace('.nebula', '.omix')

OVERWRITE = 0
TARGET = [
  'AHI',
  'age'
][0]
# -----------------------------------------------------------------------------
# (2) Load omix
# -----------------------------------------------------------------------------
assert os.path.exists(OMIX_PATH)

omix = Omix.load(OMIX_PATH)
omix.set_targets(TARGET, return_new_omix=False)
# -----------------------------------------------------------------------------
# (3) Visualization
# -----------------------------------------------------------------------------
omix.show_in_explorer()


"""
Note: 
I. omix exploration pipeline
   (1) sf_pval 1000
   (2) sf_ucp 100 0.7 
       Note 0.7 works better than 0.9, yielding final MAE as low as 7.0 (age).
   (3) (optional) sf_pval N (N < 100)
   (4) ml eln/svr/lir
"""

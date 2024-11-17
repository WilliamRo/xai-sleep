from collections import OrderedDict
from hf.sc_tools import load_macro_and_meta_from_workdir
from hf.match_lab import MatchLab
from hypnomics.hypnoprints.extractor import Extractor
from roma import finder, io

import os



# -----------------------------------------------------------------------------
# (1) Configuration
# -----------------------------------------------------------------------------
# (1.1) Working directory
WORK_DIR = r'../data/sleepedfx_sc'

# (1.2) TODO: Configure this part
CONDITIONAL = 1
PROBE_SUFFIX = ['ABC38', 'AC33', 'C31', 'AB7'][3]

# (1.3) File names
C_SUFFIX = f'{"c" if CONDITIONAL else "nc"}'

MAT_FN = f'SC-30s-{PROBE_SUFFIX}-{C_SUFFIX}.matlab'
# -----------------------------------------------------------------------------
# (2) Macro-distance omix generation
# -----------------------------------------------------------------------------
MAT_PATH = os.path.join(WORK_DIR, MAT_FN)
mat_lab = io.load_file(MAT_PATH, verbose=True)



if __name__ == '__main__':
  # PI_KEY = 'pi_test_1027'
  # PI_KEY = 'pi_test_1028_1'
  # PI_KEY = 'pi_test_1028_2'
  PI_KEY = 'pi_test_1029_1'

  mat_lab.estimate_efficacy_v1(pi_key=PI_KEY)

  io.save_file(mat_lab, MAT_PATH)

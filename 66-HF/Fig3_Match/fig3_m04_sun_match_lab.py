from roma import finder, io

import os



# -----------------------------------------------------------------------------
# (1) Configuration
# -----------------------------------------------------------------------------
# (1.1) Working directory
WORK_DIR = r'../data/sleepedfx_sc'

# (1.2) TODO: Configure this part
CONDITIONAL = 1
PROBE_SUFFIX = 'C31'

# (1.3) File names
MAT_FN = f'SC-30s-{PROBE_SUFFIX}-sun.matlab'
# -----------------------------------------------------------------------------
# (2) Macro-distance omix generation
# -----------------------------------------------------------------------------
MAT_PATH = os.path.join(WORK_DIR, MAT_FN)
mat_lab = io.load_file(MAT_PATH, verbose=True)



if __name__ == '__main__':

  # PI_KEY = 'pi_test_1027'
  # PI_KEY = 'pi_test_1028_1'
  # PI_KEY = 'pi_test_1028_2'
  # PI_KEY = 'pi_test_1029_1'

  PI_KEY = '1105v1'
  nested = 1
  if not nested: PI_KEY += '_not_nested'

  mat_lab.estimate_efficacy_v1(pi_key=PI_KEY, nested=nested,
                               plot_matrix=1, overwrite=0)

  io.save_file(mat_lab, MAT_PATH)

from roma import finder, io, console

import os



# -----------------------------------------------------------------------------
# (1) Configuration
# -----------------------------------------------------------------------------
# (1.1) Working directory
WORK_DIR = r'../data/sleepedfx_sc'

# (1.2) TODO: Configure this part
CONDITIONAL = 1
PROBE_SUFFIX = ['ABC38', 'AC33', 'C31', 'AB7', 'ABD11'][4]
INCLUDE_WAKE = 0

N_PATIENT = [71, 75][
  1]
assert N_PATIENT in (71, 75)
NP_SUFFIX = '' if N_PATIENT == 71 else '-75'

# (1.3) File names
W_SUFFIX = '' if INCLUDE_WAKE else '-NW'
C_SUFFIX = f'{"c" if CONDITIONAL else "nc"}'

MAT_FN = f'SC-30s-{PROBE_SUFFIX}-{C_SUFFIX}{W_SUFFIX}{NP_SUFFIX}.matlab'
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

  # Configure here
  M = 3
  N = 3
  ks = [400]
  ts = [0.7]
  nested = 1

  kstr = ','.join([str(k) for k in ks])
  tstr = ','.join([str(t) for t in ts])
  PI_KEY = f'M{M}N{N}ks{kstr}ts{tstr}nested{nested}'
  console.show_status(f'PI_KEY = {PI_KEY}')

  # PI_KEY = '1106v1'

  if not nested: PI_KEY += '_not_nested'


  mat_lab.estimate_efficacy_v1(pi_key=PI_KEY, nested=nested, plot_matrix=os.name == 'nt',
                               overwrite=0, ks=ks, ts=ts, M=M, N=N)

  io.save_file(mat_lab, MAT_PATH)


"""
- PI_KEY = '1105v1', nested = 1

"""

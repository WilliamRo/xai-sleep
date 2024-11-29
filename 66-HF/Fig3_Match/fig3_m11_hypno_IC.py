import sys, os

SOLUTION_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
print(f'Solution dir = {SOLUTION_DIR}')

PATH_LIST = ['66-HF', 'xai-kit', 'xai-kit/roma', 'xai-kit/pictor',
             'xai-kit/tframe']

for p in PATH_LIST: sys.path.append(os.path.join(SOLUTION_DIR, p))

from roma import finder, io, console



# -----------------------------------------------------------------------------
# (1) Configuration
# -----------------------------------------------------------------------------
# (1.1) Working directory
WORK_DIR = os.path.join(SOLUTION_DIR, '66-HF/data/sleepedfx_sc')

# (1.2) TODO: Configure this part
CONDITIONAL = 1
PROBE_SUFFIX = 'ABD11'
INCLUDE_WAKE = 0

N_PATIENT = [71, 75][
  1]
assert N_PATIENT in (71, 75)
NP_SUFFIX = '' if N_PATIENT == 71 else '-75'

# (1.3) File names
W_SUFFIX = '' if INCLUDE_WAKE else '-NW'
C_SUFFIX = f'{"c" if CONDITIONAL else "nc"}'

MAT_FN = f'SC-30s-{PROBE_SUFFIX}-{C_SUFFIX}{W_SUFFIX}{NP_SUFFIX}.matlab'

assert not INCLUDE_WAKE
assert CONDITIONAL
assert N_PATIENT == 75
# -----------------------------------------------------------------------------
# (2) Macro-distance omix generation
# -----------------------------------------------------------------------------
MAT_PATH = os.path.join(WORK_DIR, MAT_FN)
mat_lab = io.load_file(MAT_PATH, verbose=True)



if __name__ == '__main__':
  fbn = 'ST'
  assert fbn in ('ST', 'IS', 'IC')

  # Configure here
  M = 3
  N = 3
  ks = [400]
  ts = [0.7]
  nested = 1

  kstr = ','.join([str(k) for k in ks])
  tstr = ','.join([str(t) for t in ts])
  PI_KEY = f'M{M}N{N}ks{kstr}ts{tstr}nested{nested}-fbn{fbn}'
  console.show_status(f'PI_KEY = {PI_KEY}')

  # PI_KEY = '1106v1'

  if not nested: PI_KEY += '_not_nested'

  mat_lab.estimate_efficacy_v1(pi_key=PI_KEY, nested=nested, plot_matrix=1,
                               overwrite=0, ks=ks, ts=ts, M=M, N=N, fbn=fbn)

  io.save_file(mat_lab, MAT_PATH)


"""
- PI_KEY = '1105v1', nested = 1

"""

import sys, os
from tabnanny import verbose

SOLUTION_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
print(f'Solution dir = {SOLUTION_DIR}')

PATH_LIST = ['66-HF', 'xai-kit', 'xai-kit/roma', 'xai-kit/pictor',
             'xai-kit/tframe', 'hypnomics']

sys.path.append(SOLUTION_DIR)
for p in PATH_LIST: sys.path.append(os.path.join(SOLUTION_DIR, p))

from roma import finder, io, console



# -----------------------------------------------------------------------------
# (1) Configuration
# -----------------------------------------------------------------------------
# (1.1) Working directory
SRC_DIR = os.path.join(SOLUTION_DIR, '66-HF/data/sleepedfx_sc')
WORK_DIR = os.path.join(SOLUTION_DIR, '66-HF/data/match_sc')

# (1.2) TODO: Configure this part
CONDITIONAL = 1
PROBE_SUFFIX = 'Ad11'
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
MAT_PATH = os.path.join(SRC_DIR, MAT_FN)
mat_lab = io.load_file(MAT_PATH, verbose=True)

OMIX_FN = f'SC-30s-{PROBE_SUFFIX}-{C_SUFFIX}{W_SUFFIX}{NP_SUFFIX}-Dist.omix'
OMIX_PATH = os.path.join(SRC_DIR, OMIX_FN)
omix = io.load_file(OMIX_PATH, verbose=True)



if __name__ == '__main__':
  # ST: 616, IS: 462, IC: 308
  fbn = 'ST'
  assert fbn in ('ST', 'IS', 'IC')

  # Configure here
  M = 5
  N = 5
  ks = [50, 100, 200]
  ts = [0.7, 0.9]
  nested = 1

  kstr = ','.join([str(k) for k in ks])
  tstr = ','.join([str(t) for t in ts])
  PI_KEY = f'{fbn}-M{M}N{N}ks{kstr}ts{tstr}nested{nested}.pi'
  if not nested: PI_KEY += '_not_nested'

  console.show_status(f'PI_KEY = {PI_KEY}')

  PI_PATH = os.path.join(WORK_DIR, PI_KEY)
  if os.path.exists(PI_PATH):
    pi = io.load_file(PI_PATH, verbose=True)
    mat_lab.estimate_efficacy_v1(
      pi_key=pi, nested=nested, plot_matrix=os.name == 'nt',
      overwrite=0, ks=ks, ts=ts, M=M, N=N, fbn=fbn)
  else:
    pi = mat_lab.estimate_efficacy_v1(
      pi_key=PI_KEY, nested=nested, plot_matrix=os.name == 'nt',
      overwrite=0, ks=ks, ts=ts, M=M, N=N, fbn=fbn, omix=omix)
    io.save_file(pi, PI_PATH)

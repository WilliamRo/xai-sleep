import sys, os

SOLUTION_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
print(f'Solution dir = {SOLUTION_DIR}')

PATH_LIST = ['66-HF', 'xai-kit', 'xai-kit/roma', 'xai-kit/pictor',
             'xai-kit/tframe', '66-HF/Fig3_Match', 'hypnomics']

sys.path.append(SOLUTION_DIR)
for p in PATH_LIST: sys.path.append(os.path.join(SOLUTION_DIR, p))



# -----------------------------------------------------------------------------
# (1) Import
# -----------------------------------------------------------------------------
from fig3_m11_hypno_ST import mat_lab, console, io, SRC_DIR, omix, WORK_DIR



if __name__ == '__main__':
  # ST: 616, IS: 462, IC: 308
  fbn = 'IC'
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
      overwrite = 0, ks = ks, ts = ts, M = M, N = N, fbn = fbn, omix = omix)
    io.save_file(pi, PI_PATH)

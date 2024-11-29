import sys, os

SOLUTION_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
print(f'Solution dir = {SOLUTION_DIR}')

PATH_LIST = ['66-HF', 'xai-kit', 'xai-kit/roma', 'xai-kit/pictor',
             'xai-kit/tframe', '66-HF/Fig3_Match']

for p in PATH_LIST: sys.path.append(os.path.join(SOLUTION_DIR, p))



# -----------------------------------------------------------------------------
# (1) Import
# -----------------------------------------------------------------------------
from fig3_m11_hypno_ST import mat_lab, console, io, MAT_PATH



if __name__ == '__main__':
  # ST: 616, IS: 462, IC: 308
  fbn = 'IC'
  assert fbn in ('ST', 'IS', 'IC')

  # Configure here
  M = 3
  N = 3
  ks = [50, 100, 200]
  ts = [0.7, 0.9]
  nested = 1

  kstr = ','.join([str(k) for k in ks])
  tstr = ','.join([str(t) for t in ts])
  PI_KEY = f'M{M}N{N}ks{kstr}ts{tstr}nested{nested}-fbn{fbn}'
  console.show_status(f'PI_KEY = {PI_KEY}')

  if not nested: PI_KEY += '_not_nested'

  mat_lab.estimate_efficacy_v1(pi_key=PI_KEY, nested=nested, plot_matrix=1,
                               overwrite=0, ks=ks, ts=ts, M=M, N=N, fbn=fbn)

  io.save_file(mat_lab, MAT_PATH)

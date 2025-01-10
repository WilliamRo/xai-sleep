"""
Last modified: 2024-12-25
"""
# Add path in order to be compatible with Linux
import sys, os

SOLUTION_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

PATH_LIST = ['26-HSP', '26-HSP/99-data-probe', 'xai-kit', 'hypnomics', '66-HF',
             'xai-kit/roma', 'xai-kit/pictor', 'xai-kit/tframe']

if __name__ == '__main__':
  print(f'Solution dir = {SOLUTION_DIR}')
  sys.path.append(SOLUTION_DIR)
  for p in PATH_LIST: sys.path.append(os.path.join(SOLUTION_DIR, p))

from roma import io, console
from pictor.xomics.omix import Omix

import a00_common as hub



# -----------------------------------------------------------------------------
# (0) Configuration
# -----------------------------------------------------------------------------
# MAT_FN = r'HSP-100-E-6chn-30s_match_MAD1.matlab'
MAT_FN = r'HSP-100-Ab-6chn-30s_match_MAD1.matlab'
OMIX_FN = r'HSP-100-Ab-6chn-30s_match_MAD1.omix'

# -----------------------------------------------------------------------------
# (1) Load match lab and omix
# -----------------------------------------------------------------------------
MAT_PATH = os.path.join(hub.MATCH_DIR, MAT_FN)
mat_lab = io.load_file(MAT_PATH, verbose=True)
OMIX_PATH = os.path.join(hub.OMIX_DIR, OMIX_FN)
omix = Omix.load(OMIX_PATH, verbose=True)



if __name__ == '__main__':
  # Configure here
  M = 2
  N = 2
  ks = [50, 100]
  ts = [0.7, 0.8]
  nested = 1

  kstr = ','.join([str(k) for k in ks])
  tstr = ','.join([str(t) for t in ts])
  PI_KEY = f'M{M}N{N}ks{kstr}ts{tstr}nested{nested}'
  console.show_status(f'PI_KEY = {PI_KEY}')

  if not nested: PI_KEY += '_not_nested'

  mat_lab.estimate_efficacy_v1(pi_key=PI_KEY, nested=nested, omix=omix,
                               plot_matrix=os.name == 'nt',
                               overwrite=0, ks=ks, ts=ts, M=M, N=N)

  io.save_file(mat_lab, MAT_PATH)

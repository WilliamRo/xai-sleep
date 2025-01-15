"""Estimate hypnomics efficacy in age prediction.
"""
import sys, os
from tabnanny import verbose

SOLUTION_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
print(f'Solution dir = {SOLUTION_DIR}')

PATH_LIST = ['66-HF', 'xai-kit', 'xai-kit/roma', 'xai-kit/pictor',
             'xai-kit/tframe']

for p in PATH_LIST: sys.path.append(os.path.join(SOLUTION_DIR, p))

from hf.probe_tools import get_probe_keys
from pictor.xomics.omix import Omix
from pictor.xomics.evaluation.pipeline import Pipeline
from roma import console



# -----------------------------------------------------------------------------
# (1) Configuration
# -----------------------------------------------------------------------------
# (1.1) Working directory & omix path
SRC_OMIX_PATH = os.path.join(SOLUTION_DIR, r'31-OSA-XU/data/125samples-6channels-ABD-30s-NW-age.omix')
WORK_DIR = os.path.join(SOLUTION_DIR, '66-HF/data/exp2_age_rrsh')

# (1.2) TODO: Configure this part
M = 2
N = 2
# ks = [800, 1000, 1200]
# ts = [0.6, 0.7]
ks = [400, 800]
ts = [0.7, 0.9]

NESTED = 1

PLOT_MAT = os.name == 'nt'   # plot mat for windows
OVERWRITE = 0
SAVE_PKG = 1

# (1.3) MISC
OMIX_FN = f'RRSH125-30s-ABD.omix'

NESTED_SUFFIX = '_nested' if NESTED else ''
PKG_FN = f'{OMIX_FN.split(".")[0]}_M{M}N{N}_k{ks[0]}-{ks[-1]}_t{ts[0]}-{ts[-1]}{NESTED_SUFFIX}.omix'
PKG_PATH = os.path.join(WORK_DIR, PKG_FN)
# -----------------------------------------------------------------------------
# (2) Fit or load
# -----------------------------------------------------------------------------
if OVERWRITE or not os.path.exists(PKG_PATH):
  # (2.0) Load omix
  omix = Omix.load(SRC_OMIX_PATH, verbose=True)

  # TODO
  omix = omix.select_features('pval', k=3000, verbose=True)

  console.show_status(f'Feature dimension: {omix.n_features}')

  # (2.1) Initialize pipeline using macro omix
  pi = Pipeline(omix, ignore_warnings=1, save_models=1)

  # (2.2) Create subspaces
  for k, t in [(_k, _t) for _k in ks for _t in ts]: pi.create_sub_space(
    'ucp', k=k, threshold=t, repeats=M, nested=NESTED, show_progress=1)

  # # For macro features added '*'
  # pi.create_sub_space('*', repeats=M, nested=True, show_progress=1)

  # (2.3) Traverse all subspaces
  pi.fit_traverse_spaces('eln', repeats=N, nested=1, show_progress=1, verbose=0)
  # pi.fit_traverse_spaces('svr', repeats=N, nested=1, show_progress=1, verbose=0)

  # (2.4) Save packages if required
  if SAVE_PKG: omix.save(PKG_PATH, verbose=True)
else:
  # (2.a) Load omix
  omix = Omix.load(PKG_PATH)

  # (2.b) Initialize pipeline using macro omix which has been fit
  pi = Pipeline(omix, ignore_warnings=1, save_models=1)

# -----------------------------------------------------------------------------
# (3) Result analysis
# -----------------------------------------------------------------------------
# (3.1) Report results
pi.report()

if PLOT_MAT: pi.plot_matrix()




"""
"""
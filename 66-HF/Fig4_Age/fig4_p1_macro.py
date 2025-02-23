"""Estimate macro-feature efficacy in age prediction.

"""
import sys, os

SOLUTION_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
print(f'Solution dir = {SOLUTION_DIR}')

PATH_LIST = ['66-HF', 'xai-kit', 'xai-kit/roma', 'xai-kit/pictor',
             'xai-kit/tframe']

for p in PATH_LIST: sys.path.append(os.path.join(SOLUTION_DIR, p))

from pictor.xomics.omix import Omix
from pictor.xomics.evaluation.pipeline import Pipeline



# -----------------------------------------------------------------------------
# (1) Configuration
# -----------------------------------------------------------------------------
# (1.1) Working directory & omix path
MACRO_PATH = os.path.join(SOLUTION_DIR, r'66-HF/03-sleep-age/data/SC-age-macro-30.omix')
WORK_DIR = os.path.join(SOLUTION_DIR, '66-HF/data/exp2_age_sc')

# (1.2) TODO: Configure this part
M = 5
N = 5
ks = [10, 20]
ts = [0.6, 0.7, 0.8, 0.9]

PLOT_MAT = 0
OVERWRITE = 0
SAVE_PKG = 1

# (1.3) MISC
PKG_FN = f'Age_MACRO_M{M}N{N}_k{ks[0]}-{ks[-1]}_t{ts[0]}-{ts[-1]}.omix'
PKG_PATH = os.path.join(WORK_DIR, PKG_FN)
# -----------------------------------------------------------------------------
# (2) Fit or load
# -----------------------------------------------------------------------------
if OVERWRITE or not os.path.exists(PKG_PATH):
  # (2.0) Load omix
  omix = Omix.load(MACRO_PATH)
  assert omix.n_features == 30

  # (2.1) Initialize pipeline using macro omix
  pi = Pipeline(omix, ignore_warnings=1, save_models=1)

  # (2.2) Create subspaces
  for k, t in [(_k, _t) for _k in ks for _t in ts]: pi.create_sub_space(
    'ucp', k=k, threshold=t, repeats=M, nested=True, show_progress=1)

  # For macro features added '*'
  pi.create_sub_space('*', repeats=M, nested=True, show_progress=1)

  # (2.3) Traverse all subspaces
  eln_hp_space = {'alpha': [1.0], 'l1_ratio': [0.0]}
  pi.fit_traverse_spaces('eln', repeats=N, nested=1, show_progress=1,
                         verbose=0, hp_space=eln_hp_space)
  pi.fit_traverse_spaces('svr', repeats=N, nested=1, show_progress=1, verbose=0)

  # (2.4) Save packages if required
  if SAVE_PKG: omix.save(PKG_PATH, verbose=True)
else:
  # (2.a) Load omix
  omix = Omix.load(PKG_PATH)
  assert omix.n_features == 30

  # (2.b) Initialize pipeline using macro omix which has been fit
  pi = Pipeline(omix, ignore_warnings=1, save_models=1)

# -----------------------------------------------------------------------------
# (3) Result analysis
# -----------------------------------------------------------------------------
# (3.1) Report results
pi.report()

if PLOT_MAT: pi.plot_matrix(title=PKG_FN)




"""
"""

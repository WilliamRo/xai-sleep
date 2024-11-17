"""Estimate macro-feature efficacy in age prediction.

"""
from hf.probe_tools import get_probe_keys
from pictor.xomics.omix import Omix
from pictor.xomics.evaluation.pipeline import Pipeline, FitPackage

import os



# -----------------------------------------------------------------------------
# (1) Configuration
# -----------------------------------------------------------------------------
# (1.1) Working directory & omix path
SRC_OMIX_DIR = r'../data/sleepedfx_sc'
WORK_DIR = r'../data/exp2_age_sc'

# (1.2) TODO: Configure this part
M = 2
N = 2
ks = [20, 35, 50]
ts = [0.5, 0.7, 0.9]

CONDITIONAL = 1
PROBE_CONFIG = 'ABD'
INCLUDE_MACRO = 0
INCLUDE_WAKE = 0
NESTED = 0

PLOT_MAT = 1
OVERWRITE = 0
SAVE_PKG = 1

# (1.3) MISC
C_SUFFIX = f'{"c" if CONDITIONAL else "nc"}'
PROBE_KEYS = get_probe_keys(PROBE_CONFIG)
PROBE_SUFFIX = f'{PROBE_CONFIG}{len(PROBE_KEYS)}'
MACRO_SUFFIX = f'-MACRO' if INCLUDE_MACRO else ''
OMIX_FN = f'SC-30s-{PROBE_SUFFIX}-{C_SUFFIX}{MACRO_SUFFIX}.omix'

NESTED_SUFFIX = '_nested' if NESTED else ''
PKG_FN = f'{OMIX_FN.split(".")[0]}_M{M}N{N}_k{ks[0]}-{ks[-1]}_t{ts[0]}-{ts[-1]}{NESTED_SUFFIX}.omix'
PKG_PATH = os.path.join(WORK_DIR, PKG_FN)
# -----------------------------------------------------------------------------
# (2) Fit or load
# -----------------------------------------------------------------------------
if OVERWRITE or not os.path.exists(PKG_PATH):
  # (2.0) Load omix
  OMIX_PATH = os.path.join(SRC_OMIX_DIR, OMIX_FN)

  omix = Omix.load(OMIX_PATH)
  if not INCLUDE_WAKE: omix = omix.filter_by_name('W', include=False)

  # (2.1) Initialize pipeline using macro omix
  pi = Pipeline(omix, ignore_warnings=1, save_models=1)

  # (2.2) Create subspaces
  for k, t in [(_k, _t) for _k in ks for _t in ts]: pi.create_sub_space(
    'ucp', k=k, threshold=t, repeats=M, nested=NESTED, show_progress=1)

  # # For macro features added '*'
  # pi.create_sub_space('*', repeats=M, nested=True, show_progress=1)

  # (2.3) Traverse all subspaces
  pi.fit_traverse_spaces('eln', repeats=N, nested=1, show_progress=1, verbose=0)
  pi.fit_traverse_spaces('svr', repeats=N, nested=1, show_progress=1, verbose=0)

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

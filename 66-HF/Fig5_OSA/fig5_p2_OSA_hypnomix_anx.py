from pictor.xomics.omix import Omix
from pictor.xomics.evaluation.pipeline import Pipeline, FitPackage

import os



# -----------------------------------------------------------------------------
# (1) Configuration
# -----------------------------------------------------------------------------
# (1.1) Nebula configuration
SRC_DIR = r'../../31-OSA-XU/data'  # contains cloud files
SAVE_DIR = r'data'

# (1.2) TODO
OVERWRITE = 0
TARGET = 'anx'
M = 3
N = 3
ks = [50, 100, 150]
ts = [0.6, 0.8]
n_folds = 2

NESTED = 1
PLOT_MAT = 1
SUFFIX = ''

# (1.3) Set path
OMIX_FN = '125samples-6channels-ABD-30s.omix'
OMIX_PATH = os.path.join(SRC_DIR, OMIX_FN)

NESTED_SUFFIX = 'nested' if NESTED else 'xnested'
PKG_FN = f'125samples-6channels-ABD-30s-anx90-{NESTED_SUFFIX}{SUFFIX}.omix'
PKG_PATH = os.path.join(SAVE_DIR, PKG_FN)

# -----------------------------------------------------------------------------
# (2) Pipeline
# -----------------------------------------------------------------------------
if OVERWRITE or not os.path.exists(PKG_PATH):
  omix = Omix.load(OMIX_PATH)
  omix = omix.set_targets(TARGET, return_new_omix=True)
  omix = omix.select_features('pval', k=1000)

  # (2.1) Initialize pipeline
  pi = Pipeline(omix, ignore_warnings=1, save_models=1)

  # (2.2) Create subspaces
  for k, t in [(_k, _t) for _k in ks for _t in ts]: pi.create_sub_space(
    'ucp', k=k, threshold=t, repeats=M, nested=NESTED, show_progress=1)

  # (2.3) Traverse all subspaces
  pi.fit_traverse_spaces('lr', repeats=N, nested=NESTED, show_progress=1,
                         verbose=0, n_splits=n_folds)
  pi.fit_traverse_spaces('svm', repeats=N, nested=NESTED, show_progress=1,
                         verbose=0, n_splits=n_folds)

  omix.save(PKG_PATH, verbose=True)
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

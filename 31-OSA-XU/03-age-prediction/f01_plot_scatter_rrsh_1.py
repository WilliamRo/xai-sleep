from pictor.xomics.omix import Omix
from pictor.xomics.evaluation.pipeline import Pipeline, FitPackage
from pictor.xomics.evaluation.reg_ana import RegressionAnalysis
from roma import console

import matplotlib.pyplot as plt
import numpy as np
import sys, os



# -----------------------------------------------------------------------------
# (1) Configuration
# -----------------------------------------------------------------------------
SOLUTION_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

# (1.1) Working directory & omix path
WORK_DIR = os.path.join(SOLUTION_DIR, '66-HF/data/exp2_age_rrsh')

# (1.2) TODO: Configure here
PKG_FN, SF_KEY, ML_KEY = [
  ('RRSH125-30s-ABD_M2N2_k800-1200_t0.6-0.7_nested.omix', 'ucp-800-t0.7', 'ELN'),
  ('RRSH125-30s-ABD_M2N2_k400-800_t0.7-0.9_nested.omix', 'ucp-800-t0.7', 'ELN'),
][1]

PKG_PATH = os.path.join(WORK_DIR, PKG_FN)

# -----------------------------------------------------------------------------
# (2) Load data
# -----------------------------------------------------------------------------
omix = Omix.load(PKG_PATH)

pi = Pipeline(omix, ignore_warnings=1, save_models=1)
pi.report()

# -----------------------------------------------------------------------------
# (3) Define functions
# -----------------------------------------------------------------------------
row_labels, col_labels, matrix_dict = pi.get_pkg_matrix(abbreviate=True)
assert SF_KEY in row_labels and ML_KEY in col_labels

pkg_list = matrix_dict[(SF_KEY, ML_KEY)]
pkg_list = sorted(pkg_list, key=lambda x: x['mae'])

MAEs = [pkg['mae'] for pkg in pkg_list]
MAE_str = [f'{v:.2f}' for v in MAEs]
console.show_status(f'MAEs = [{", ".join(MAE_str)}]')

def get_best_CA_BA():
  pkg: FitPackage = pkg_list[0]
  ca = pkg.targets
  ba = pkg.probabilities

  return ca, ba

def get_avg_CA_BA():
  ca_list, ba_list = [], []
  for pkg in pkg_list:
    sorted_targets, sorted_probs = pkg.ordered_targets_and_probs
    ca_list.append(sorted_targets)
    ba_list.append(sorted_probs)

  ca = ca_list[0]
  assert all([np.allclose(ca, _ca) for _ca in ca_list])
  ba = np.average(ba_list, axis=0)

  return ca, ba

# -----------------------------------------------------------------------------
# (4) Plot
# -----------------------------------------------------------------------------
LEGEND_FONT_SIZE = 10

# ca, ba = get_best_CA_BA()
ca, ba = get_avg_CA_BA()

ra = RegressionAnalysis(ca, ba)

ra.plot_scatter(
  figsize=(4.5, 3.5),
  xlabel='Chronological Age (yr)',
  ylabel='Estimated Age (yr)',
  upper_left='B',
  upper_left_x=-0.13,
)


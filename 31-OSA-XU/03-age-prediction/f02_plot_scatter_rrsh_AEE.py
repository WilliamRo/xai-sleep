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

# Get sorted targets and probabilities
ca_list, ba_list = [], []
for pkg in pkg_list:
  sorted_targets, sorted_probs = pkg.ordered_targets_and_probs
  ca_list.append(sorted_targets)
  ba_list.append(sorted_probs)

ca = ca_list[0]
assert all([np.allclose(ca, _ca) for _ca in ca_list])
ba = np.average(ba_list, axis=0)

# -----------------------------------------------------------------------------
# (4) Plot
# -----------------------------------------------------------------------------
import matplotlib.pyplot as plt

LEGEND_FONT_SIZE = 10
FIG_SIZE = (4.5, 3.5)

# Instantiate an ra
ra = RegressionAnalysis(ca, ba)

fig = plt.figure(figsize=FIG_SIZE)
ax: plt.Axes = fig.add_subplot(1, 1, 1)

# Plot ideal line
min_y, max_y = min(ra.true_y), max(ra.true_y)
ax.plot([min_y, max_y], [min_y, max_y], 'r--', label='Identity line')

# Plot data
label = f'MAE = {ra.mae:.2f}'
lo, hi = ra.r_CI95
label += f'\nr = {ra.r:.2f} (CI95% = [{lo:.2f}, {hi:.2f}])'
ax.plot(ra.true_y, ra.pred_y, 'o', label=label, alpha=0.8)

# Set x, y labels
xlabel = 'Chronological Age (yr)',
ylabel = 'Estimated Age (yr)',
ax.set_xlabel(xlabel), ax.set_ylabel(ylabel)

ax.legend(fontsize=LEGEND_FONT_SIZE)
ax.grid(True)

# Upper left symbol
symbol = 'B'
ax.text( -0.13, 1.0, symbol, transform=ax.transAxes, fontsize=16,
         fontweight='bold', va='top', ha='left')

plt.tight_layout()
plt.show()

from scipy.spatial import distance_matrix

from pictor.xomics.omix import Omix
from pictor.xomics.evaluation.pipeline import Pipeline, FitPackage
from pictor.xomics.evaluation.reg_ana import RegressionAnalysis
from roma import console
from roma import finder, io
from hf.match_lab import MatchLab

import matplotlib.pyplot as plt
import numpy as np
import sys, os



# -----------------------------------------------------------------------------
# (1) Configuration
# -----------------------------------------------------------------------------
# (1.1) Working directory
WORK_DIR = r'../data/sleepedfx_sc'

# (1.2) TODO: Configure this part
CONDITIONAL = 1
PROBE_SUFFIX = ['ABC38', 'AC33', 'C31', 'AB7', 'ABD11'][4]
INCLUDE_WAKE = 0
SF_KEY, ML_KEY = 'ucp-400-t0.7', 'LoR'

N_PATIENT = [71, 75][
  1]
assert N_PATIENT in (71, 75)
NP_SUFFIX = '' if N_PATIENT == 71 else '-75'

# (1.3) File names
W_SUFFIX = '' if INCLUDE_WAKE else '-NW'
C_SUFFIX = f'{"c" if CONDITIONAL else "nc"}'

MAT_FN = f'SC-30s-{PROBE_SUFFIX}-{C_SUFFIX}{W_SUFFIX}{NP_SUFFIX}.matlab'
# -----------------------------------------------------------------------------
# (2) Load pipeline, get pkg list
# -----------------------------------------------------------------------------
MAT_PATH = os.path.join(WORK_DIR, MAT_FN)
mat_lab: MatchLab = io.load_file(MAT_PATH, verbose=True)

M = 3
N = 3
ks = [400]
ts = [0.7]
nested = 1

kstr = ','.join([str(k) for k in ks])
tstr = ','.join([str(t) for t in ts])
PI_KEY = f'M{M}N{N}ks{kstr}ts{tstr}nested{nested}'
console.show_status(f'PI_KEY = {PI_KEY}')
# PI_KEY = '1106v1'

pi: Pipeline = mat_lab.get_from_pocket(PI_KEY)

row_labels, col_labels, matrix_dict = pi.get_pkg_matrix(abbreviate=True)
pkg_list = matrix_dict[(SF_KEY, ML_KEY)]
pkg_list = sorted(pkg_list, key=lambda x: x['AUC'])

AUCs = [pkg['AUC'] for pkg in pkg_list]
AUC_str = [f'{v:.2f}' for v in AUCs]
console.show_status(f'MAEs = [{", ".join(AUC_str)}]')

# -----------------------------------------------------------------------------
# (3) Calculate and plot
# -----------------------------------------------------------------------------
def get_avg_distmat():
  d_list = []
  for pkg in pkg_list:
    sample_labels = pkg.get_from_pocket('sample_labels', key_should_exist=True)
    sample_labels = list(sample_labels)
    sorted_sample_labels = sorted(sample_labels)
    sorted_indices = [sample_labels.index(i) for i in sorted_sample_labels]
    d_list.append(pkg.probabilities[sorted_indices])

  d = np.average(d_list, axis=0)

  N = int(np.sqrt(len(d)))
  m = np.zeros((N, N), dtype=float)
  for i in range(N):
    for j in range(N):
      index = sorted_sample_labels.index(f'({i+1}, {j+1})')
      m[i, j] = d[index][1]

  return m

m = get_avg_distmat()

# Using MatchLab.analyze can show Top-1/5 Acc
# mat_lab.analyze(distance_matrix=m, matrices=[], labels=[])
# quit()

# Plot
from mpl_toolkits.axes_grid1 import make_axes_locatable

fig = plt.figure(figsize=(5, 5))
ax: plt.Axes = fig.add_subplot(1, 1, 1)

cmap = ['hot'][0]
im = ax.imshow(m, cmap=cmap, interpolation='none', vmin=0, vmax=1)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
cbar = fig.colorbar(im, cax=cax)
cbar.set_label('Distance')

ax.set_xlabel('Second Night')
ax.set_ylabel('First Night')
ax.set_xticks([])
ax.set_yticks([])

plt.tight_layout()
plt.show()

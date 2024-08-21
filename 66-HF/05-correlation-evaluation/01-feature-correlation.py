from collections import OrderedDict
from hypnomics.freud.nebula import Nebula
from hf.model_helper import gen_dist_mat
from hf.sc_tools import get_dual_nebula, get_joint_key
from hf.sc_tools import CK_MAP, PK_MAP
from mpl_toolkits.axes_grid1 import make_axes_locatable
from roma import console
from roma import finder
from x_dual_view import PAIRED_LABELS

import os
import numpy as np



# -----------------------------------------------------------------------------
# (1) Configuration
# -----------------------------------------------------------------------------
WORK_DIR = r'../data/sleepedfx_sc'
# SG_LABELS = ['SC4001E', 'SC4002E']
SG_LABELS = finder.walk(WORK_DIR, type_filter='dir', return_basename=True)[:999]

# [ 2, 5, 10, 30, ]
TIME_RESOLUTION = 30

# NEB_FN = f'SC-{TIME_RESOLUTION}-KDE.nebula'
NEB_FN = f'SC-{TIME_RESOLUTION}-KDE-0730.nebula'
neb_file_path = os.path.join(WORK_DIR, NEB_FN)

# -----------------------------------------------------------------------------
# (2) Load nebula
# -----------------------------------------------------------------------------
nebula: Nebula = Nebula.load(neb_file_path)

# -----------------------------------------------------------------------------
# (3) Analyze correlation
# -----------------------------------------------------------------------------
feature_dict = OrderedDict()

label = nebula.labels[1]

# for pk in nebula.probe_keys:
#   feature_dict[pk] = []
#   for ck in nebula.channels:
#     for sk in ('W', 'N1', 'N2', 'N3', 'R'):
#       feature_dict[pk].extend(nebula.data_dict[(label, ck, pk)][sk])

# feature_labels = [PK_MAP[pk] for pk in feature_dict.keys()]

CHANNELS = [
  'EEG Fpz-Cz',
  'EEG Pz-Oz'
]

PROBE_KEYS = [
  'FREQ-20',
  'AMP-1',
  # 'GFREQ-35',
  # 'P-TOTAL',
  'RP-DELTA', 'RP-THETA', 'RP-ALPHA',
  'RP-BETA',
]

CHNL_PROB_KEYS = [(ck, pk) for ck in CHANNELS for pk in PROBE_KEYS]

for ck, pk in CHNL_PROB_KEYS:
  feature_dict[(ck, pk)] = []
  for sk in ('W', 'N1', 'N2', 'N3', 'R'):
    feature_dict[(ck, pk)].extend(nebula.data_dict[(label, ck, pk)][sk])

feature_labels = [f'{CK_MAP[ck]}-{PK_MAP[pk]}'
                  for ck, pk in feature_dict.keys()]

# -----------------------------------------------------------------------------
# (4) Plot
# -----------------------------------------------------------------------------
import matplotlib.pyplot as plt

features = np.stack(feature_dict.values())

matrix = np.corrcoef(features, rowvar=True)

# Plot
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(matrix, cmap='RdBu', vmin=-1, vmax=1)

plt.title(label)

# Set ticks
show_name = True
if show_name:
  ticks = np.arange(len(feature_labels))
  ax.set_yticks(ticks=ticks, labels=feature_labels)
  ax.set_xticks(ticks=ticks, labels=feature_labels)
  fig.autofmt_xdate(rotation=45)

# Show color bar
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im, cax=cax)

d = 3
for i in range(matrix.shape[0]):
  for j in range(matrix.shape[1]):
    text = f'{matrix[i, j]:.{d}f}'
    color = 'black' if -0.5 < matrix[i, j] < 0.5 else 'white'
    ax.text(j, i, text, ha='center', va='center', fontsize=8, color=color)

# Show
fig.tight_layout()
plt.show()


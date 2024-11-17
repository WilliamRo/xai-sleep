"""A figure of 2D.
X-axis, AUC for subject matching performance.
Y-axis, ranking of (wo upsilon) -> (w upsilon), sorted by difference

Data for plotting is in 66-HF/
"""
from hf.probe_tools import get_probe_keys
from roma import io

import matplotlib.pyplot as plt
import os



# -----------------------------------------------------------------------------
# (1) Configuration
# -----------------------------------------------------------------------------
# (1.1) Target nebula file path
WORK_DIR = r'./data'
SAVE_PATH = os.path.join(WORK_DIR, 'c_nc_auc.od')
# [ 2(x), 5(x), 10(x), 30, ]
TIME_RESOLUTION = 30
NEB_FN = f'SC-{TIME_RESOLUTION}s-KDE-39-probes.nebula'
neb_file_path = os.path.join(WORK_DIR, NEB_FN)

# (1.2) Set channels
CHANNELS = [
  'EEG Fpz-Cz',
  'EEG Pz-Oz'
]

# (1.3) Set probe keys
PROBE_CONFIG = 'ABD'
PROBE_KEYS = get_probe_keys(PROBE_CONFIG)

# -----------------------------------------------------------------------------
# (2) Read data
# -----------------------------------------------------------------------------
od = io.load_file(SAVE_PATH)

# -----------------------------------------------------------------------------
# (3) Plot
# -----------------------------------------------------------------------------
# (3.1) Generate orders
def probe_score(pk):
  return max(od[(CHANNELS[0], pk, 1)], od[(CHANNELS[1], pk, 1)])

probe_keys = sorted(PROBE_KEYS, key=probe_score)

# (3.3) Configuration
delta = 0.1
ms = 8
colors = ['#af4141', '#3b6ea9']

# (3.3) Generate orders
fig = plt.figure(figsize=(9, 6))
ax: plt.Axes = fig.add_subplot(1, 1, 1)
for i, pk in enumerate(probe_keys):
  y = i

  for j, ck in enumerate(CHANNELS):
    yj = y - (j - 0.5) * 2 * delta
    auc_nc = od[(ck, pk, 0)]
    auc_c = od[(ck, pk, 1)]

    color = colors[j]

    label = f'{ck}' if i == 0 else None
    ax.plot([auc_nc, auc_c], [yj, yj], color=color, label=label)

    ax.plot(auc_nc, yj, 'o', markeredgecolor=color, markersize=ms,
            markerfacecolor='white')
    ax.plot(auc_c, yj, 'o', markeredgecolor=color, markersize=ms,
            markerfacecolor=color)

# Draw split lines
x0, x1 = ax.get_xlim()
for i, pk in enumerate(probe_keys):
  if i == 0: continue
  # Get current limits
  ax.plot([x0, x1], [i - 0.5, i - 0.5], color='grey', linestyle=':')

ax.set_xlabel('AUC')
ax.set_yticks(list(range(len(probe_keys))), probe_keys)
ax.set_xlim([x0, x1])
ax.legend()

plt.tight_layout()
plt.show()

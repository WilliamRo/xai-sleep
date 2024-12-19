from collections import OrderedDict
from freud.hypno_tools.probe_tools import get_probe_keys
from hypnomics.freud.freud import Freud
from hypnomics.freud.nebula import Nebula
from hypnomics.freud.telescopes.telescope import Telescope
from hypnomics.freud.telescopes.popglass import PopGlass
from roma import console, finder

import os



# -----------------------------------------------------------------------------
# (1) Configuration
# -----------------------------------------------------------------------------
# (1.1) Configure directories
NEB_DIR = r'../data/sleepedfx_sc'
POP_DIR = r'../../data/sleepedfx-sc/sc_population'

# (1.2) Fixed configuration, do not modify
TIME_RESOLUTION = 30

CHANNELS = [
  'EEG Fpz-Cz',
  'EEG Pz-Oz'
]

# (1.3) TODO: configure here
# Ab: 'POWER-30', 'PR-DELTA_TOTAL', 'PR-THETA_TOTAL', 'PR-ALPHA_TOTAL',
#     'PR-BETA_TOTAL', 'PR-SIGMA_TOTAL', 'PR-DELTA_THETA', 'PR-DELTA_ALPHA',
#     'PR-THETA_ALPHA', 'FREQ-20', 'AMP-1'
PROBE_CONFIG = 'Ab'
PROBE_KEYS = get_probe_keys(PROBE_CONFIG, expand_group=True)

# (1.*) Auto configuration
NEB_FN = f'SC-153-{PROBE_CONFIG}-{len(CHANNELS)}chn-{TIME_RESOLUTION}s.nebula'
# NEB_FN = 'SC-153-Ab-2chn-30s.nebula'   # 1
NEB_PATH = os.path.join(POP_DIR, NEB_FN)
# -----------------------------------------------------------------------------
# (2) Manage nebula library
# -----------------------------------------------------------------------------
assert os.path.exists(NEB_PATH)
nebula: Nebula = Nebula.load(NEB_PATH, verbose=True)

# -----------------------------------------------------------------------------
# (3) Merge
# -----------------------------------------------------------------------------
label_age_tuples = [(lb, nebula.meta[lb]['age']) for lb in nebula.labels]
age_list = [age for _, age in label_age_tuples]
min_age, max_age = min(age_list), max(age_list)

console.show_status(f'Age range: [{min_age}, {max_age}]')

SPAN = 10
STRIDE = 5
AGE_INTERVALS = [(a, a + SPAN) for a in range(min_age, min_age + (max_age - min_age) // SPAN * SPAN + SPAN, STRIDE)]
ALIGN = ['none', 'stage', 'channel_stage'][1]

od = OrderedDict()
for ai in AGE_INTERVALS:
  od[ai] = [lb for lb, age in label_age_tuples if ai[0] <= age <= ai[1]]

console.show_status(f'{sum([len(lb_list) for lb_list in od.values()])} subjects included.')
psg_labels = []
for ai, lb_list in od.items():
  N = len(lb_list)
  console.supplement(f'{ai}: {N}')
  label = f'({ai[0]},{ai[1]}), N={N}'

  if N == 0: continue
  psg_labels.append(label)
  nebula.merge(lb_list, merged_label=label, save_to_self=True, align=ALIGN)

nebula.set_labels(psg_labels, check_sub_set=False)
# -----------------------------------------------------------------------------
# (4) Visualize nebula
# -----------------------------------------------------------------------------
import re
import matplotlib.pyplot as plt

PK1, PK2 = 'FREQ-20', 'AMP-1'
# PK1, PK2 = 'POWER-30', 'PR-THETA_ALPHA'

def bottom_plotter(ax: plt.Axes, label):
  # (1) Extract age1 and age2 from e.g., "(age1, age2), N=30"
  a1, a2 = [float(a) for a in re.search(r'\((\d+),(\d+)\)', label).groups()]
  # (2) Show rectangle
  ax.add_patch(plt.Rectangle((a1, 0), a2 - a1, 1, color='red', alpha=0.2))
  # (3) Set style
  ax.set_yticks([])
  ax.set_ylim([0, 1])
  ax.set_xlim([AGE_INTERVALS[0][0], AGE_INTERVALS[-1][-1]])

nebula.dual_view(x_key=PK1, y_key=PK2, viewer_class=PopGlass, bottom_ratio=0.05,
                 bottom_plotter=bottom_plotter, outlier_coef=0.5)

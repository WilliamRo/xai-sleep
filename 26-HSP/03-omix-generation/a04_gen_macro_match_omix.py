"""
Last modified: 2024-12-25

This script is for generating macro omix given nebula.
Ref: 66-HF/Fig3_Match/fig3_a01_macro_omix.py
"""
# Add path in order to be compatible with Linux
import sys, os

SOLUTION_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

PATH_LIST = ['26-HSP', '26-HSP/99-data-probe', 'xai-kit', 'hypnomics', '66-HF',
             'xai-kit/roma', 'xai-kit/pictor', 'xai-kit/tframe']

if __name__ == '__main__':
  print(f'Solution dir = {SOLUTION_DIR}')
  sys.path.append(SOLUTION_DIR)
  for p in PATH_LIST: sys.path.append(os.path.join(SOLUTION_DIR, p))

from collections import OrderedDict
from freud.talos_utils.sleep_sets.hsp import HSPAgent
from hypnomics.freud.nebula import Nebula
from pictor.xomics.omix import Omix
from roma import io

import a00_common as hub
import numpy as np
import re



# -----------------------------------------------------------------------------
# (0) Configuration
# -----------------------------------------------------------------------------
OVERWRITE = 0

MAD = 1

INCLUDE_TRANSITION_PER_HOUR = False
assert not INCLUDE_TRANSITION_PER_HOUR

NEB_FN = f'HSP-100-E-6chn-30s.nebula'

OMIX_FN = re.search(r'HSP-\d+', NEB_FN).group(0) + f'-macro_match_MAD{MAD}.omix'
OMIX_PATH = os.path.join(hub.OMIX_DIR, OMIX_FN)

# -----------------------------------------------------------------------------
# (1) Load or crate .omix
# -----------------------------------------------------------------------------
if os.path.exists(OMIX_PATH) and not OVERWRITE:
  omix: Omix = Omix.load(OMIX_PATH, verbose=True)
else:
  from hf.match_lab import MatchLab

  # (1.1) Load nebula
  NEB_PATH = os.path.join(hub.OMIX_DIR, NEB_FN)
  nebula: Nebula = Nebula.load(NEB_PATH, verbose=True)
  SG_LABELS = nebula.labels

  # (1.2) Split nebula
  neb_1, neb_2 = HSPAgent.get_dual_nebula(nebula, max_age_diff=MAD)

  # (1.3) Load features
  x_dict = {}
  for pid in SG_LABELS:
    macro_path = os.path.join(hub.CLOUD_DIR, pid, 'macro_alpha.od')
    x_dict[pid] = io.load_file(macro_path)

  for k, v in x_dict.items():
    v.pop('Transition_per_Hour')
    assert len(v) == 30

  # (1.4) Read macro features
  F1, F2 = OrderedDict(), OrderedDict()
  for pid in neb_1.labels: F1[pid] = x_dict[pid]
  for pid in neb_2.labels: F2[pid] = x_dict[pid]

  # (2.4) Instantiate a match-lab
  mat_lab = MatchLab(F1, F2)
  omix = mat_lab.get_pair_omix(k=99999)

  # (2.5) Save data
  omix.save(OMIX_PATH, verbose=True)

# -----------------------------------------------------------------------------
# (2) Show in explorer
# -----------------------------------------------------------------------------
if not hub.IN_LINUX: omix.show_in_explorer()

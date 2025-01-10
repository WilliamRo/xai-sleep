"""
Last modified: 2024-12-25
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

from freud.hypno_tools.probe_tools import get_probe_keys
from freud.talos_utils.sleep_sets.hsp import HSPSet, HSPAgent
from hypnomics.freud.nebula import Nebula
from hypnomics.hypnoprints.extractor import Extractor
from pictor.xomics.omix import Omix
from roma import io, console

import a00_common as hub
import numpy as np



# -----------------------------------------------------------------------------
# (0) Configuration
# -----------------------------------------------------------------------------
OVERWRITE = 0

MAD = 0.2

# NEB_FN = f'HSP-100-Ab-6chn-30s.nebula'
NEB_FN = f'HSP-378-Ab-6chn-30s.nebula'
# NEB_FN = f'HSP-218-Ab-2chn-30s.nebula'
OMIX_FN = NEB_FN.replace('.nebula', f'_match_MAD{MAD}.omix')
# OMIX_FN = NEB_FN.replace('.nebula', f'_match_MAD{MAD}_.omix')
MAT_FN = OMIX_FN.replace('.omix', '.matlab')

# -----------------------------------------------------------------------------
# (1) Generate match omix
# -----------------------------------------------------------------------------
OMIX_PATH = os.path.join(hub.OMIX_DIR, OMIX_FN)
MAT_PATH = os.path.join(hub.MATCH_DIR, MAT_FN)

if os.path.exists(OMIX_PATH) and not OVERWRITE:
  omix: Omix = Omix.load(OMIX_PATH, verbose=True)
else:
  from hf.match_lab import MatchLab

  # (1.1) Load nebula
  NEB_PATH = os.path.join(hub.OMIX_DIR, NEB_FN)
  nebula: Nebula = Nebula.load(NEB_PATH, verbose=True)

  # (1.2) Split nebula
  neb_1, neb_2 = HSPAgent.get_dual_nebula(nebula, max_age_diff=MAD)

  # (1.3) Extract features
  extractor = Extractor()
  F1 = extractor.extract(neb_1, return_dict=True)
  F2 = extractor.extract(neb_2, return_dict=True)

  # (1.4) Instantiate a match-lab and save
  mat_lab = MatchLab(F1, F2)
  io.save_file(mat_lab, MAT_PATH)

  omix = mat_lab.get_pair_omix(k=99999)
  omix = omix.filter_by_name('W_', include=False)
  omix.save(OMIX_PATH, verbose=True)



if __name__ == '__main__':
  SHUFFLE = 0
  ICC = 0.6

  if not hub.IN_LINUX:
    # Randomly shuffle targets if required
    if SHUFFLE:
      import numpy as np
      _tgts = list(omix.targets)
      np.random.shuffle(_tgts)
      omix.targets = np.array(_tgts)

    if ICC > 0:
      indices = []
      for i, fl in enumerate(omix.feature_labels):
        icc = float(fl.split('=')[1][:-1])
        if icc > ICC: indices.append(i)

      omix = omix.get_sub_space(indices, start_from_1=False)

    omix.show_in_explorer()

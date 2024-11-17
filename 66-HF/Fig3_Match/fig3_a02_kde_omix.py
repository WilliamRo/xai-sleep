from collections import OrderedDict
from hf.match_lab import MatchLab
from hf.probe_tools import get_probe_keys
from pictor.xomics.omix import Omix
from roma import console, finder, io

import numpy as np
import os



# -----------------------------------------------------------------------------
# (1) Configuration
# -----------------------------------------------------------------------------
# (1.1) Working directory & patient inclusion
WORK_DIR = r'../data/sleepedfx_sc'
SG_LABELS = finder.walk(WORK_DIR, type_filter='dir', return_basename=True)

# (1.2) TODO: Configure this part
CONDITIONAL = 1
PROBE_CONFIG = 'ABC'
OVERWRITE = 0

CHANNELS = [
  'EEG Fpz-Cz',
  'EEG Pz-Oz'
]

# (1.3) File names and MISC
C_SUFFIX = f'{"c" if CONDITIONAL else "nc"}'

PROBE_KEYS = get_probe_keys(PROBE_CONFIG)
PROBE_SUFFIX = f'{PROBE_CONFIG}{len(PROBE_KEYS)}'

MAT_FN = f'SC-30s-KDE-{PROBE_SUFFIX}-{C_SUFFIX}.matlab'
DIST_OMIX_FN = f'SC-30s-KDE-{PROBE_SUFFIX}-{C_SUFFIX}-Dist.omix'

KDE_DIST_FN = f'SC-30s-ABC38.kdist'
KDE_DIST_PATH = os.path.join(WORK_DIR, KDE_DIST_FN)

N = 71
# -----------------------------------------------------------------------------
# (2) Macro-distance omix generation
# -----------------------------------------------------------------------------
DIST_OMIX_PATH = os.path.join(WORK_DIR, DIST_OMIX_FN)
MAT_PATH = os.path.join(WORK_DIR, MAT_FN)

if os.path.exists(DIST_OMIX_PATH) and not OVERWRITE:
  omix = Omix.load(DIST_OMIX_PATH)
else:
  SAVE_DATA = True

  if os.path.exists(MAT_PATH) and not OVERWRITE:
    mat_lab = io.load_file(MAT_PATH)
  else:
    # (2.1) Read kde_dist_dict_repo
    kde_dist_dict_repo = io.load_file(KDE_DIST_PATH)

    # (2.2) Construct object for initializing MatchLab
    CHNL_PROB_KEYS = [(ck, pk) for ck in CHANNELS for pk in PROBE_KEYS]
    kde_dict_for_matlab = OrderedDict()
    nights_1, nights_2 = None, None
    for ck, pk in CHNL_PROB_KEYS:
      # (2.2.1) Check matrix key
      mat_key = (ck, pk, CONDITIONAL)
      if mat_key not in kde_dist_dict_repo:
        SAVE_DATA = False
        continue
      kde_dist_dict = kde_dist_dict_repo[mat_key]
      if len(kde_dist_dict) != N * N:
        SAVE_DATA = False
        continue

      # (2.2.2) Generate nights_1 and nights_2 if not exist
      if nights_1 is None:
        nights_1 = list(sorted(set([k[0] for k in kde_dist_dict.keys()])))
        nights_2 = list(sorted(set([k[1] for k in kde_dist_dict.keys()])))
        # Sanity check
        assert len(nights_1) == N and len(nights_2) == N
        for n1, n2 in zip(nights_1, nights_2): assert n1[:5] == n2[:5]

      # (2.2.3) Generate distance matrix
      dist_mat = np.zeros((N, N), dtype=np.float32)
      for i, n1 in enumerate(nights_1):
        for j, n2 in enumerate(nights_2):
          d = kde_dist_dict[(n1, n2)]
          dist_mat[i, j] = d

      # (2.2.4) Set distance matrix to kde_dict_for_matlab
      kde_dict_for_matlab[f'{ck}-{pk}-{C_SUFFIX}'] = dist_mat

    # (2.3) Instantiate a match-lab and save
    mat_lab = MatchLab(kde_dist_dict=kde_dict_for_matlab)
    if SAVE_DATA: io.save_file(mat_lab, MAT_PATH, verbose=True)

  # (2.4) Generate omix and save
  omix = mat_lab.get_pair_omix(k=99999)
  if SAVE_DATA: omix.save(DIST_OMIX_PATH)
  else: console.warning(f'`{DIST_OMIX_FN}` not saved due to missing data')



if __name__ == '__main__':
  omix.show_in_explorer()

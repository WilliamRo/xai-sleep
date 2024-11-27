"""Nebula is generated using fig0_c_nc_neb_gen.py
"""
from pygments.lexer import include

from hf.match_lab import MatchLab
from hf.probe_tools import get_probe_keys
from hypnomics.hypnoprints.extractor import Extractor
from pictor.xomics.omix import Omix
from roma import finder, io

import os



# -----------------------------------------------------------------------------
# (1) Configuration
# -----------------------------------------------------------------------------
# (1.1) Working directory & patient inclusion
WORK_DIR = r'../data/sleepedfx_sc'

# (1.2) TODO: Configure this part
CONDITIONAL = 1
PROBE_CONFIG = 'ABD'
OVERWRITE = 0
INCLUDE_WAKE = 0

N_PATIENT = [71, 75][1]
assert N_PATIENT in (71, 75)
NP_SUFFIX = '' if N_PATIENT == 71 else '-75'

# (1.3) File names and MISC
C_SUFFIX = f'{"c" if CONDITIONAL else "nc"}'
W_SUFFIX = '' if INCLUDE_WAKE else '-NW'

PROBE_KEYS = get_probe_keys(PROBE_CONFIG)
PROBE_SUFFIX = f'{PROBE_CONFIG}{len(PROBE_KEYS)}'

# This nebula file includes all subjects' info
NEB_FN = f'SC-30s-ABC38.nebula'

MAT_FN = f'SC-30s-{PROBE_SUFFIX}-{C_SUFFIX}{W_SUFFIX}{NP_SUFFIX}.matlab'
DIST_OMIX_FN = f'SC-30s-{PROBE_SUFFIX}-{C_SUFFIX}{W_SUFFIX}{NP_SUFFIX}-Dist.omix'
# -----------------------------------------------------------------------------
# (2) Macro-distance omix generation
# -----------------------------------------------------------------------------
DIST_OMIX_PATH = os.path.join(WORK_DIR, DIST_OMIX_FN)
MAT_PATH = os.path.join(WORK_DIR, MAT_FN)

if os.path.exists(DIST_OMIX_PATH) and not OVERWRITE:
  omix = Omix.load(DIST_OMIX_PATH)
else:
  if os.path.exists(MAT_PATH) and not OVERWRITE:
    mat_lab = io.load_file(MAT_PATH)
  else:
    # (2.1) Read nebula
    from hf.sc_tools import get_dual_nebula
    from hypnomics.freud.nebula import Nebula
    from x_dual_view import PAIRED_LABELS, PAIRED_LABELS_ALL

    nebula: Nebula = Nebula.load(os.path.join(WORK_DIR, NEB_FN))
    nebula.set_probe_keys(PROBE_KEYS)
    nebula.set_labels(PAIRED_LABELS)
    neb_1, neb_2 = get_dual_nebula(nebula)

    # (2.2) Extract features
    if CONDITIONAL:
      E_SETTINGS = {
        'include_statistical_features': 1,
        'include_inter_stage_features': 1,
        'include_inter_channel_features': 1,

        'include_proportion': 0,
        'include_stage_shift': 0,
        'include_channel_shift': 0,
        'include_stage_wise_covariance': 0,
        'include_stage_mean': 0,
        'include_all_mean_std': 0,
      }
    else:
      # TODO:
      E_SETTINGS = {
        'include_statistical_features': 0,
        'include_inter_stage_features': 0,
        'include_inter_channel_features': 0,

        'include_proportion': 0,
        'include_stage_shift': 0,
        'include_channel_shift': 0,
        'include_stage_wise_covariance': 0,
        'include_stage_mean': 0,
        'include_all_mean_std': 1,
      }
    extractor = Extractor(**E_SETTINGS)
    F1 = extractor.extract(neb_1, return_dict=True)
    F2 = extractor.extract(neb_2, return_dict=True)

    # (2.3) Instantiate a match-lab and save
    mat_lab = MatchLab(F1, F2)
    io.save_file(mat_lab, MAT_PATH)

  # (2.4) Generate omix and save
  omix = mat_lab.get_pair_omix(k=99999)
  if not INCLUDE_WAKE: omix = omix.filter_by_name('W', include=False)
  omix.save(DIST_OMIX_PATH)



if __name__ == '__main__':
  omix.show_in_explorer()

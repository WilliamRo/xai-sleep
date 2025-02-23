"""Experiment 1: individual matching
"""
from collections import OrderedDict
from hf.sc_tools import load_macro_and_meta_from_workdir
from hf.match_lab import MatchLab
from pictor.xomics.omix import Omix
from roma import finder

import os



# -----------------------------------------------------------------------------
# (1) Configuration
# -----------------------------------------------------------------------------
WORK_DIR = r'../data/sleepedfx_sc'
SG_LABELS = finder.walk(WORK_DIR, type_filter='dir', return_basename=True)

XLSX_PATH = r'../../data/sleep-edf-database-expanded-1.0.0/SC-subjects.xls'

MACRO_DIST_OMIX_FN = 'macro_dist_30.omix'
MACRO_DIST_OMIX_PATH = os.path.join(WORK_DIR, MACRO_DIST_OMIX_FN)

OVERWRITE = 0
# -----------------------------------------------------------------------------
# (2) Macro-distance omix generation
# -----------------------------------------------------------------------------

if os.path.exists(MACRO_DIST_OMIX_PATH) and not OVERWRITE:
  macro_dist_omix = Omix.load(MACRO_DIST_OMIX_PATH)
else:
  # (2.1) Read nebula, the content is not important
  from hf.sc_tools import get_dual_nebula
  from hypnomics.freud.nebula import Nebula
  from x_dual_view import PAIRED_LABELS

  NEB_FN = f'SC-30-KDE-0730.nebula'
  nebula: Nebula = Nebula.load(os.path.join(WORK_DIR, NEB_FN))
  nebula.set_labels(PAIRED_LABELS)
  neb_1, neb_2 = get_dual_nebula(nebula)

  # (2.2) Read macro features
  x_dict, meta_dict = load_macro_and_meta_from_workdir(
    WORK_DIR, SG_LABELS, XLSX_PATH)

  for k, v in x_dict.items():
    v.pop('Transition_per_Hour')
    assert len(v) == 30

  # (2.3) Read macro features
  F1, F2 = OrderedDict(), OrderedDict()
  for pid in neb_1.labels: F1[pid] = x_dict[pid]
  for pid in neb_2.labels: F2[pid] = x_dict[pid]

  # (2.4) Instantiate a match-lab
  mat_lab = MatchLab(F1, F2)
  macro_dist_omix = mat_lab.get_pair_omix(k=99999)

  # (2.5) Save omix
  macro_dist_omix.save(MACRO_DIST_OMIX_PATH)



if __name__ == '__main__':
  macro_dist_omix.show_in_explorer()

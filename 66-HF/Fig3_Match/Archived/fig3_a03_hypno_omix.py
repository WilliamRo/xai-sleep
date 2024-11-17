from collections import OrderedDict
from hf.sc_tools import load_macro_and_meta_from_workdir
from hf.match_lab import MatchLab
from hypnomics.hypnoprints.extractor import Extractor
from pictor.xomics.omix import Omix
from roma import finder

import os



# -----------------------------------------------------------------------------
# (1) Configuration
# -----------------------------------------------------------------------------
WORK_DIR = r'../../data/sleepedfx_sc'
SG_LABELS = finder.walk(WORK_DIR, type_filter='dir', return_basename=True)

XLSX_PATH = r'../../../data/sleep-edf-database-expanded-1.0.0/SC-subjects.xls'

DIST_OMIX_FN = 'hypno-39.omix'
DIST_OMIX_PATH = os.path.join(WORK_DIR, DIST_OMIX_FN)
# -----------------------------------------------------------------------------
# (2) Macro-distance omix generation
# -----------------------------------------------------------------------------
if os.path.exists(DIST_OMIX_PATH):
  omix = Omix.load(DIST_OMIX_PATH)
else:
  # (2.1) Read nebula, the content is not important
  from hf.sc_tools import get_dual_nebula
  from hypnomics.freud.nebula import Nebula
  from x_dual_view import PAIRED_LABELS

  NEB_FN = f'SC-30s-KDE-39-probes.nebula'
  nebula: Nebula = Nebula.load(os.path.join(WORK_DIR, NEB_FN))
  nebula.set_labels(PAIRED_LABELS)
  neb_1, neb_2 = get_dual_nebula(nebula)

  # (2.2) Read features
  extractor = Extractor()
  F1 = extractor.extract(neb_1, return_dict=True)
  F2 = extractor.extract(neb_2, return_dict=True)

  # (2.4) Instantiate a match-lab
  mat_lab = MatchLab(F1, F2)
  omix = mat_lab.get_pair_omix(k=99999)

  # (2.5) Save omix
  omix.save(DIST_OMIX_PATH)



if __name__ == '__main__':
  omix.show_in_explorer()

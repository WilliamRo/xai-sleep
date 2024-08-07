from hypnomics.hypnoprints.extractor import Extractor
from hypnomics.freud.nebula import Nebula
from hf.sc_tools import get_paired_sg_labels, get_dual_nebula
from hf.match_lab import MatchLab
from pictor.xomics.omix import Omix
from pictor.xomics.evaluation.pipeline import Pipeline, FitPackage
from roma import finder
from roma import console

import os
import numpy as np



# -----------------------------------------------------------------------------
# (1) Configuration
# -----------------------------------------------------------------------------
from x_dual_view import configs, WORK_DIR, CHANNELS, PK1, PK2, SG_LABELS, TIME_RESOLUTION, NEB_FN, PAIRED_LABELS

NEB_FN = f'SC-153-partial-{TIME_RESOLUTION}.nebula'
# -----------------------------------------------------------------------------
# (2) Get dual nebula
# -----------------------------------------------------------------------------
nebula: Nebula = Nebula.load(os.path.join(WORK_DIR, NEB_FN))
nebula.set_labels(PAIRED_LABELS)

neb_1, neb_2 = get_dual_nebula(nebula)
# -----------------------------------------------------------------------------
# (3) Extract features
# -----------------------------------------------------------------------------
extractor = Extractor()
F1 = extractor.extract(neb_1, return_dict=True)
F2 = extractor.extract(neb_2, return_dict=True)

# -----------------------------------------------------------------------------
# (4) Pair-wise learning
# -----------------------------------------------------------------------------
matlab = MatchLab(F1, F2, normalize=1, N=999,
                  neb_1=neb_1, neb_2=neb_2, nebula=nebula)

k = 2
OMIX_NAME = f'0628_SC150_K{k}.omix'
OMIX_SAVE_PATH = os.path.join(WORK_DIR, OMIX_NAME)

if os.path.exists(OMIX_SAVE_PATH):
  omix = Omix.load(OMIX_SAVE_PATH)
  pi = Pipeline(omix, ignore_warnings=1, save_models=1)
else:
  omix = matlab.get_pair_omix(k=k)
  # omix.show_in_explorer()
  # exit()

  pi = matlab.fit_pipeline(omix, show_progress=1)
  omix.save(OMIX_SAVE_PATH, verbose=True)

pi.report()
# pi.plot_matrix()

# -----------------------------------------------------------------------------
# (5) Validation
# -----------------------------------------------------------------------------
omix_val = matlab.get_pair_omix(k=50)
pkg: FitPackage = pi.evaluate_best_pipeline(omix_val)
pkg.report()
matlab.dm_validate(pi, ranking=3, reducer=None)



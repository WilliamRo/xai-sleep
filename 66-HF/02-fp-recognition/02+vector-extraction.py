"""

"""
from hypnomics.hypnoprints.extractor import Extractor
from hypnomics.freud.nebula import Nebula
from hf.sc_tools import get_paired_sg_labels, get_dual_nebula
from hf.match_lab import MatchLab
from roma import finder
from roma import console

import os


# -----------------------------------------------------------------------------
# (1) Configuration
# -----------------------------------------------------------------------------
from x_dual_view import configs, WORK_DIR, CHANNELS, PK1, PK2, SG_LABELS, TIME_RESOLUTION, NEB_FN, PAIRED_LABELS

TIME_RESOLUTION = 30

# NEB_FN = f'SC-153-partial-{TIME_RESOLUTION}.nebula'
NEB_FN = f'SC-{TIME_RESOLUTION}-KDE-0730.nebula'
# NEB_FN = f'SC-{TIME_RESOLUTION}s-KDE-39-probes.nebula'
# -----------------------------------------------------------------------------
# (2) Get dual nebula
# -----------------------------------------------------------------------------
nebula: Nebula = Nebula.load(os.path.join(WORK_DIR, NEB_FN))
nebula.set_labels(PAIRED_LABELS)

neb_1, neb_2 = get_dual_nebula(nebula)

# viewer_class = Telescope
# neb_2.dual_view(x_key=PK1, y_key=PK2, viewer_class=viewer_class, **configs)

# -----------------------------------------------------------------------------
# (3) Extract features and analysis
# -----------------------------------------------------------------------------
extractor = Extractor()
F1 = extractor.extract(neb_1, return_dict=True)
F2 = extractor.extract(neb_2, return_dict=True)

matlab = MatchLab(F1, F2, normalize=1, N=999,
                  neb_1=neb_1, neb_2=neb_2, nebula=nebula)

matlab.select_feature(min_ICC=0.5, verbose=1, set_C=1)
if 0:
  # Show ICC plot
  # matlab.ICC_analysis(ymax=30)

  matlab.analyze(toolbar=1)
  exit()

if 1:
  omix = matlab.get_pair_omix(k=999, include_dm=1)
  omix.show_in_explorer()
  exit()


import os

from hypnomics.hypnoprints.extractor import Extractor
from hypnomics.freud.nebula import Nebula
from hf.sc_tools import get_paired_sg_labels, get_dual_nebula
from hf.match_lab import MatchLab
from roma import finder
from roma import console



# -----------------------------------------------------------------------------
# (1) Configuration
# -----------------------------------------------------------------------------
configs = {
  # 'xmin': 3, 'xmax': 9, 'ymin': 0, 'ymax': 200,
  'show_kde': 0,
  'show_scatter': 0,
  'show_vector': 1,
  # 'scatter_alpha': 0.05,
}

WORK_DIR = r'../data/sleepedfx_sc'
CHANNELS = [
  'EEG Fpz-Cz',
  'EEG Pz-Oz'
]
PK1 = 'FREQ-20'
PK2 = 'AMP-1'

# SG_LABELS = ['SC4001E', 'SC4002E']
SG_LABELS = finder.walk(WORK_DIR, type_filter='dir', return_basename=True)[:999]
PAIRED_LABELS = get_paired_sg_labels(SG_LABELS)

# [ 2, 5, 10, 30, ]
TIME_RESOLUTION = 10

NEB_FN = 'SC-153-partial.nebula'
# -----------------------------------------------------------------------------
# (2) Get dual nebula
# -----------------------------------------------------------------------------
nebula: Nebula = Nebula.load(os.path.join(WORK_DIR, NEB_FN))
nebula = nebula[PAIRED_LABELS]

neb_1, neb_2 = get_dual_nebula(nebula)

# viewer_class = Telescope
# neb_2.dual_view(x_key=PK1, y_key=PK2, viewer_class=viewer_class, **configs)

# -----------------------------------------------------------------------------
# (3) Extract features and analysis
# -----------------------------------------------------------------------------
extractor = Extractor()
F1 = extractor.extract(neb_1, return_dict=True)
F2 = extractor.extract(neb_2, return_dict=True)

matlab = MatchLab(F1, F2, 20)

matlab.analyze()


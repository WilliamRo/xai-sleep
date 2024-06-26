import os

from hypnomics.freud.nebula import Nebula
from hypnomics.freud.telescopes.telescope import Telescope
from hf.sc_tools import get_paired_sg_labels
from roma import finder



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

WORK_DIR = r'../../data/sleepedfx_sc'
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
# (2) Visualize
# -----------------------------------------------------------------------------
nebula: Nebula = Nebula.load(os.path.join(WORK_DIR, NEB_FN))
nebula = nebula[PAIRED_LABELS]

viewer_class = Telescope
# viewer_class = None
nebula.dual_view(x_key=PK1, y_key=PK2, viewer_class=viewer_class, **configs)

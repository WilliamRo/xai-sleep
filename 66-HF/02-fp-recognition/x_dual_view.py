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
  'show_kde': 1,
  'show_scatter': 0,
  'show_vector': 0,
  # 'scatter_alpha': 0.05,
}

WORK_DIR = r'../data/sleepedfx_sc'
CHANNELS = [
  'EEG Fpz-Cz',
  'EEG Pz-Oz'
]
PK1 = 'FREQ-20'
PK2 = 'AMP-1'

SG_LABELS = finder.walk(WORK_DIR, type_filter='dir', return_basename=True)[:999]
EXCLUDES = ('422', '459', '476', '411')
PAIRED_LABELS = get_paired_sg_labels(SG_LABELS, excludes=EXCLUDES)
PAIRED_LABELS_ALL = get_paired_sg_labels(SG_LABELS)

# [ 2, 5, 10, 30, ]
TIME_RESOLUTION = 30

# NEB_FN = f'SC-153-partial-{TIME_RESOLUTION}.nebula'
NEB_FN = f'SC-{TIME_RESOLUTION}s-KDE-39-probes.nebula'
# -----------------------------------------------------------------------------
# (2) Visualize
# -----------------------------------------------------------------------------
if __name__ == '__main__':
  nebula: Nebula = Nebula.load(os.path.join(WORK_DIR, NEB_FN))
  nebula.set_labels(PAIRED_LABELS)

  viewer_class = Telescope
  nebula.dual_view(x_key=PK1, y_key=PK2, viewer_class=viewer_class, **configs)

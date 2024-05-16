import numpy as np

from roma import console
from sc.sc_agent import SCAgent
from scipy import stats



# ----------------------------------------------------------------------------
# Load valid fp_dict
# ----------------------------------------------------------------------------
sca = SCAgent(data_key='dual;alpha')
sca.report_data_info()

# ----------------------------------------------------------------------------
# Analyze features
# ----------------------------------------------------------------------------
channels = ['EEG Fpz-Cz', 'EEG Pz-Oz']
max_freqs = [20, 30]

# Calculate distance matrices
N = 67
for i, pid in enumerate(sca.beta_uni_subs[:N]):
  console.supplement(f'{i+1:02d}/{N:02d} {pid}')

matrices, labels = [], []
for chn in channels:
  dim2_tuple = ('BM02-AMP', 'pool_size', 128)
  for mf in max_freqs:
    dim1_tuple = ('BM01-FREQ', 'max_freq', mf)
    matrix = sca.get_distance_matrix(chn, dim1_tuple, dim2_tuple, N)
    matrices.append(matrix)
    labels.append(f'{chn} (F{mf})')

# Visualize distance matrices
from pictor import Pictor
p = Pictor.image_viewer('Hypno Analysis')
p.plotters[0].set('cmap', 'RdYlGn')
p.plotters[0].set('color_bar', True)
p.plotters[0].set('title', True)
p.objects = matrices
p.labels = labels
p.show()



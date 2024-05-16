from roma import console
from sc.sc_agent import SCAgent
from scipy import stats

import matplotlib.pyplot as plt
import numpy as np



# ----------------------------------------------------------------------------
# Load valid fp_dict
# ----------------------------------------------------------------------------
sca = SCAgent(data_key='dual;alpha')
sca.report_data_info()

# ----------------------------------------------------------------------------
# Configure
# ----------------------------------------------------------------------------
channel = ['EEG Fpz-Cz', 'EEG Pz-Oz']
max_freq = [20, 30][0]

dim1_tuple = ('BM01-FREQ', 'max_freq', max_freq)
dim2_tuple = ('BM02-AMP', 'pool_size', 128)

N = 67
# ----------------------------------------------------------------------------
# Fit
# ----------------------------------------------------------------------------
M = sca.get_distance_matrix(channel, dim1_tuple, dim2_tuple, N, version='v2')

acc = np.mean(np.argmin(M, axis=1) == np.arange(N))

# Do t-test
# Get diagonal elements
inner = M[np.diag_indices(N)]
# Get non-diagonal elements
inter = M[~np.eye(N, dtype=bool)]
t_stat, p_val = stats.ttest_ind(
  inner, inter[:N], equal_var=False, alternative='less')
# t_stat, p_val = stats.mannwhitneyu( inner, inter[:N], alternative='less')
# ----------------------------------------------------------------------------
# Visualization
# ----------------------------------------------------------------------------
from pictor import Pictor
p = Pictor.image_viewer('Matrices')
# p.plotters[0].set('vmin', 0)
# p.plotters[0].set('vmax', 1)
p.plotters[0].set('cmap', 'plasma')
p.plotters[0].set('color_bar', True)
p.plotters[0].set('title', True)
p.objects = [M]
p.labels = [f'{N} subjects, Accuracy = {acc:.3f}, P = {p_val}']
p.show()




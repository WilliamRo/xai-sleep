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
channel = ['EEG Fpz-Cz', 'EEG Pz-Oz'][0]
max_freq = [20, 30][0]
dim1_tuple = ('BM01-FREQ', 'max_freq', max_freq)
dim2_tuple = ('BM02-AMP', 'pool_size', 128)

# (1) Generate KDE
console.show_status('Estimating kde vectors ...')
kde_dicts_1, kde_dicts_2 = sca.get_kde_dicts(channel, dim1_tuple, dim2_tuple)

assert len(kde_dicts_1) == len(kde_dicts_2)
console.show_status(f'Successfully generated {len(kde_dicts_1)} kde vectors')

# (2) Calculate features
fea_dicts_1, fea_dicts_2 = {}, {}
console.show_status('Extracting features ...')
for s in kde_dicts_1.keys():
  fea_dicts_1[s] = sca.pyhypnomics(kde_dicts_1[s])
  fea_dicts_2[s] = sca.pyhypnomics(kde_dicts_2[s])

# (3) Calculate and visualize distance matrix
N = 20
uni_subs = sca.beta_uni_subs[:N]
matrix = np.zeros((1, N, N))
for i, s1 in enumerate(uni_subs):
  for j, s2 in enumerate(uni_subs):
    matrix[0, i, j] = np.linalg.norm(fea_dicts_1[s1] - fea_dicts_2[s2])

from pictor import Pictor
p = Pictor.image_viewer('Hypno Analysis')
p.plotters[0].set('cmap', 'RdYlGn')
p.plotters[0].set('color_bar', True)
p.objects = matrix
p.show()



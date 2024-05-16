import numpy as np

from roma import console
from sc.sc_agent import SCAgent
from sc.tf_optimizer import MatOptimizer



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
N = 67
lr = 0.01
top_K = 1
max_steps = 2001
p_cycle = 100
feature_version = 'v2'

dim1_tuple = ('BM01-FREQ', 'max_freq', max_freq)
dim2_tuple = ('BM02-AMP', 'pool_size', 128)

for i, pid in enumerate(sca.beta_uni_subs[:N]):
  console.supplement(f'{i+1:02d}/{N:02d} {pid}')
# ----------------------------------------------------------------------------
# Analyze features
# ----------------------------------------------------------------------------
# Fetch data
fea_dicts_1, fea_dicts_2 = sca.get_feature_dicts(
  channel, dim1_tuple, dim2_tuple, version=feature_version)
uni_subs = sca.beta_uni_subs[:N]

# Optimization
F1 = np.vstack([fea_dicts_1[s] for s in uni_subs])
F2 = np.vstack([fea_dicts_2[s] for s in uni_subs])

optimizer = MatOptimizer(F1, F2, lr=lr, top_K=top_K)
losses, accs, mats, steps = optimizer.fit(
  max_steps=max_steps, print_cycle=p_cycle, print_w=0)

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
p.objects = mats
p.labels = [f'[{s}/{max_steps}] Loss={l:.3f}, Acc={a:.3f}'
            for l, a, s in zip(losses, accs, steps)]
p.show()



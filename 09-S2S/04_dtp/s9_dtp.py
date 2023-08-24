import sys
sys.path.append('../')
sys.path.append('../../')

from tframe.utils.script_helper import Helper
s = Helper()

from s2s_core import th
s.register_flags(type(th))
# -----------------------------------------------------------------------------
# Configure data set here
# -----------------------------------------------------------------------------
pass

# -----------------------------------------------------------------------------
# Specify summary file name and GPU ID here
# -----------------------------------------------------------------------------
summ_name = s.default_summ_name
gpu_id = 0

s.register('gather_summ_name', summ_name + '.sum')
s.register('gpu_id', gpu_id)
s.register('gpu_memory_fraction', 0.4)
s.register('allow_growth', False)
# -----------------------------------------------------------------------------
# Set up your models and run
# -----------------------------------------------------------------------------
s.register('train', True)
s.register('epoch', 100000)
s.register('patience', 50)

# (1) Data space
channels = ['EEGx2,EOGx1']
data_configs = [f'sleepeason1 {c} alpha pattern=.*(sleepedfx)'
                for c in channels]
s.register('data_config', data_configs)
s.register('epoch_pad', 0, 1)

# (2) Model space
s.register('activation', 'lrelu', 'relu')
s.register('dtp_en_filters', 32, 64, 128)
s.register('dtp_en_ks', 16, 32, 64)

s.register('filters', 64, 128)
s.register('kernel_size', 3, 7, 11)
s.register('dtpM', 2, 3, 4)
s.register('dtpR', 2, 3)

# (3) Optimization space
s.register('global_l2_penalty', 0.001, 0.01)

# s.register('optimizer', 'adam', 'sgd')
s.register('lr', 0.0005, 0.003)
s.register('batch_size', 64, 256)


s.configure_engine(strategy='skopt', criterion='Best F1')
s.run(rehearsal=0)

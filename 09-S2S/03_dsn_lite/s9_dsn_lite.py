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

channels = ['EEGx1', 'EEGx2', 'EEGx1,EOGx1', 'EEGx2,EOGx1']
data_configs = [f'sleepeason1 {c} alpha pattern=.*(sleepedfx)'
                for c in channels]
s.register('data_config', data_configs)

s.register('use_batchnorm', s.true_and_false)
s.register('dropout', 0.3, 0.5, 0.7)
s.register('global_l2_penalty', 0.0, 0.001, 0.01)

# s.register('optimizer', 'adam', 'sgd')
s.register('lr', 0.0001, 0.0003, 0.001)

s.register('batch_size', 32, 64, 128, 256)


s.configure_engine(strategy='skopt', criterion='Best F1')
s.run(rehearsal=0)

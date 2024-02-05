import sys
sys.path.append('../')
sys.path.append('../../')

from tframe.utils.script_helper import Helper
s = Helper()

from rnn_core import th
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
s.register('patience', 30)

s.register('filters', 32, 64, 128)
s.register('use_rnn', 0, 1)

channels = ['EEGx2,EOGx1']
data_configs = [f'sleepeasonx {c} beta' for c in channels]
s.register('data_config', data_configs)

s.register('lr', 0.0001)
s.register('batch_size', 32)
s.register('num_steps', 20)
# s.register('lr', 0.0005, 0.003)
# s.register('batch_size', 5, 20)
# s.register('num_steps', 5, 50)

# s.configure_engine(strategy='skopt', criterion='Best F1')
s.configure_engine(times=5)
s.run(rehearsal=0)

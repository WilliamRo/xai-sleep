import sys
sys.path.append('../')
sys.path.append('../../')

from tframe.utils.script_helper import Helper
s = Helper()

from attn_core import th
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
s.register('gpu_memory_fraction', 0.9)
s.register('allow_growth', False)
# -----------------------------------------------------------------------------
# Set up your models and run
# -----------------------------------------------------------------------------
s.register('train', True)
s.register('epoch', 100000)
s.register('patience', 50)

channels = ['EEGx2,EOGx1']
data_configs = [f'sleepeasonx {c} beta pattern=.*(sleepedfx)'
                for c in channels]
s.register('data_config', data_configs)
s.register("sg_buffer_size", 120)
s.register('epoch_pad', 0, 1, 2, 3, 4, 5)

s.register('use_batchnorm', s.true)
s.register('dropout', 0.5)
s.register('global_l2_penalty', 0.001, 0.01)

# s.register('optimizer', 'adam', 'sgd')
s.register('lr', 0.0005, 0.003)

s.register('batch_size', 256, 512)


s.configure_engine(strategy='skopt', criterion='Best F1')
s.run(rehearsal=0)


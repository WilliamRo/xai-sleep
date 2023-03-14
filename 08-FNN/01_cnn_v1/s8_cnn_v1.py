import sys
sys.path.append('../')
sys.path.append('../../')

from tframe.utils.script_helper import Helper
s = Helper()

from fnn_core import th
s.register_flags(type(th))
# -----------------------------------------------------------------------------
# Configure data set here
# -----------------------------------------------------------------------------
s.register('gpu_memory_fraction', 0.7)

# -----------------------------------------------------------------------------
# Specify summary file name and GPU ID here
# -----------------------------------------------------------------------------
gpu_id = 0
summ_name = s.default_summ_name

s.register('gather_summ_name', summ_name + '.sum')
s.register('gpu_id', gpu_id)
# -----------------------------------------------------------------------------
# Specified hyper-parameters
# -----------------------------------------------------------------------------
s.register('epoch', 10000)
s.register('patience', 30)
s.register('batch_size', 32, 64, 128)
s.register('lr', 0.0001, 0.0003, 0.0008)
s.register('use_batchnorm', s.true_and_false)

s.register('epoch_delta', 0.0, 0.1, 0.2, 0.3)
# -----------------------------------------------------------------------------
# Set up your models and run
# -----------------------------------------------------------------------------

s.configure_engine(times=5)
s.run()

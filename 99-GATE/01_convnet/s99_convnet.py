import sys
sys.path.append('../')
sys.path.append('../../')

from tframe.utils.script_helper import Helper
s = Helper()

from gate_core import th
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
s.register('allow_growth', False)
# -----------------------------------------------------------------------------
# Set up your models and run
# -----------------------------------------------------------------------------
s.register('train', True)
s.register('epoch', 1000)
s.register('patience', 15)

# s.register('ratio', *[0.1*i for i in range(11)])
data_configs = ['sleepedf:20:0,1', 'sleepedf:20:0,2', 'sleepedf:20:1,2']
s.register('data_config', *data_configs)
test_configs = [f'test_data:{2*i},{2*i+1}' for i in range(10)]
s.register('test_config', *test_configs)
s.configure_engine(times=5)
s.run(rehearsal=False)
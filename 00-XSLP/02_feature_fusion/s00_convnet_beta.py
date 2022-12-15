import sys
sys.path.append('../')
sys.path.append('../../')

from tframe.utils.script_helper import Helper
s = Helper()

from xslp_core import th
s.register_flags(type(th))
# -----------------------------------------------------------------------------
# Configure data set here
# -----------------------------------------------------------------------------
pass

# -----------------------------------------------------------------------------
# Specify summary file name and GPU ID here
# -----------------------------------------------------------------------------
summ_name = '1215_s00_feature_fusion_convnet'
gpu_id = 0

s.register('gather_summ_name', summ_name + '.sum')
s.register('gpu_id', gpu_id)
s.register('allow_growth', True)
# -----------------------------------------------------------------------------
# Set up your models and run
# -----------------------------------------------------------------------------
s.register('train', True)
s.register('epoch', 1000)
s.register('patience', 15)

data_configs = ['sleepedf:20:0,2,4', 'sleepedf:20:0,1,2']
s.register('data_config', *data_configs)
test_configs = [f'test_data:{2*i},{2*i+1}' for i in range(10)]
s.register('test_config', *test_configs)
s.configure_engine(times=3)
s.run(rehearsal=False)

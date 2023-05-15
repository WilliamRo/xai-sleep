from freud.talos_utils.slp_agent import SleepAgent
from freud.talos_utils.slp_set import DataSet
from s2s_core import th



th.data_config = 'sleepedfx 1,2'
th.data_config += ' val_ids=12,13,14,15 test_ids=16,17,18,19'

ds, _, _ = SleepAgent.load_data()

for batch in ds.gen_batches(32, is_training=True):
  print()
  break

print()

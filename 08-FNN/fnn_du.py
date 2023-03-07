from freud.talos_utils.slp_agent import SleepAgent
from freud.talos_utils.slp_set import SleepSet
from freud.talos_utils.slp_set import DataSet



def load_data():
  from fnn_core import th

  ds = SleepAgent.load_data()

  return ds



if __name__ == '__main__':
  from fnn_core import th

  th.data_config = 'sleepedfx 1,2'

  ds = load_data()
  val_set: DataSet = ds.validation_set

  th.balance_classes = True
  for data in val_set.gen_batches(20, shuffle=True, is_training=True):
    print()




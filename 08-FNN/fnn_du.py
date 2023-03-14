from freud.talos_utils.slp_agent import SleepAgent
from freud.talos_utils.slp_set import SleepSet
from freud.talos_utils.slp_set import DataSet



def load_data():
  from fnn_core import th

  data_sets = SleepAgent.load_data()

  js = [1, 2]
  # if th.epoch_delta == 0: js.append(0)
  for j in js: data_sets[j] = data_sets[j].validation_set

  return data_sets



if __name__ == '__main__':
  from fnn_core import th

  th.data_config = 'sleepedfx 1,2'
  th.data_config += ' val_ids=12,13,14,15 test_ids=16,17,18,19'
  # th.data_config += ' preprocess=iqr mad=10'

  train_set, _, _ = load_data()

  assert isinstance(train_set, DataSet)

  for batch in train_set.gen_batches(50, is_training=True):
    print()





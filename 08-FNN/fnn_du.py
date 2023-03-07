from freud.talos_utils.slp_agent import SleepAgent
from freud.talos_utils.slp_set import SleepSet
from freud.talos_utils.slp_set import DataSet



def load_data():
  from fnn_core import th

  data_sets = SleepAgent.load_data()

  return data_sets



if __name__ == '__main__':
  from fnn_core import th

  th.data_config = 'sleepedfx 1,2'
  th.data_config = 'sleepedfx 1,2 valids=12,13,14,15 testids=16,17,18,19'

  ds = load_data()
  th.balance_classes = True




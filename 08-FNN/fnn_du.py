from freud.talos_utils.slp_agent import SleepAgent
from freud.talos_utils.slp_set import SleepSet



def load_data():
  from fnn_core import th

  SleepAgent.load_data()



if __name__ == '__main__':
  from fnn_core import th

  th.data_config = 'sleepedfx'

  _ = load_data()



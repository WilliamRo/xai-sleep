from slp_agent import SLPAgent

def load_data():
  # Load data
  # ...
  # train_set, val_set, test_set = SleepRecord.load(configure=configure)
  train_set, val_set, test_set = SLPAgent.load(configure=None)
  return train_set, val_set, test_set


def configure(data_set):
  return data_set


if __name__ == '__main__':
  train_set, val_set, test_set = load_data()

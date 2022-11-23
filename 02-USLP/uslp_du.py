from slp_agent import SLPAgent
from roma import console

import uslp_core as core
import numpy as np

def load_data():
  # Load data
  # ...
  train_set, val_set, test_set = SLPAgent.load(configure=None, format=format)
  return train_set, val_set, test_set


def configure(data_set, **kwargs):
  return data_set

def format(data_set):
  """
  :param data_set:
  :return:
     features:[(n,105000,c), (n,105000,c),...]  #105000=35*3000
     targets:[(n,35), (n,35),...]
  """
  from tframe import hub as th

  console.show_status(f'Formating data...')
  features = data_set.features
  targets = data_set.targets
  window_size = th.window_size
  for i, sg_targets in enumerate(targets):
    sg_targets = sg_targets[:len(sg_targets)//th.window_size*th.window_size]
    targets_reshape = np.asarray(np.split(sg_targets, len(sg_targets)//th.window_size))
    targets[i] = targets_reshape
  for i, sg_features in enumerate(features):
    sg_features = sg_features[:len(targets[i])*th.window_size*th.random_sample_length]
    features_reshape = np.asarray(np.split(sg_features, len(targets[i])))
    features[i] = features_reshape

  # Set features
  data_set.features = features
  data_set.targets = targets
  console.show_status(f'Finishing formating data...')

  return data_set

if __name__ == '__main__':
  train_set, val_set, test_set = load_data()

  # Initiate a pictor
  # p = pictor(title='sleep monitor', figure_size=(15, 9))

  # set plotter
  # m = monitor()
  # p.add_plotter(m)

  # set objects
  # p.objects = sleep_data_list

  # Begin main loop
  # p.show()

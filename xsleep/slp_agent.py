from tframe.data.base_classes import DataAgent
from tframe import console
from xsleep.slp_set import SleepSet

import os
import pickle



class SLPAgent(DataAgent):
  """To use this class, th.data_config must be specified.
  The syntax is `<...>:<...>`, e.g.,
  """

  @classmethod
  def load(cls, **kwargs):
    """th.data_config syntax:
        data_name:data_num
    """
    from xslp_core import th

    # Find 51-SLEEP/data/sleepedfx
    dataset_name, data_num = th.data_config.split(':')

    person_num = f'(all)' if data_num == '' else f'({data_num})'
    prefix = dataset_name + person_num
    suffix = 'cnn'
    if th.use_rnn:
      suffix = 'rnn'
    tfd_format_path = os.path.join(th.data_dir, dataset_name,
                                       f'{prefix}-format-{suffix}.tfds')

    if os.path.exists(tfd_format_path):
      with open(tfd_format_path,'rb') as _input_:
        console.show_status('Loading `{}` ...'.format(tfd_format_path))
        dataset = pickle.load(_input_)
    else:
      tfd_config_path = os.path.join(th.data_dir, dataset_name,
                                     f'{prefix}-config-{suffix}.tfds')
      if os.path.exists(tfd_config_path):
        with open(tfd_config_path,'rb') as _input_:
          console.show_status(f'loading {tfd_config_path}...')
          dataset = pickle.load(_input_)
      else:
        dataset = cls.load_as_tframe_data(data_dir=th.data_dir,
                                          data_name=dataset_name,
                                          first_k=data_num,
                                          suffix='-alpha')

        # Configure dataset (put this block into right position)
        configure = kwargs.get('configure', None)
        if callable(configure):
          dataset = configure(dataset)
        else:
          dataset.configure(channel_select='0,1,2')
        dataset.save(tfd_config_path)
      dataset.format_data()
      dataset.save(tfd_format_path)
      console.show_status(f'Saving {tfd_format_path} ...')

    # Convert format of label to one-hot
    import numpy as np
    label_dict = [[1,0,0,0,0], [0,1,0,0,0], [0,0,1,0,0], [0,0,0,1,0],
                  [0,0,0,0,1]]
    targets = dataset.targets
    if th.use_rnn is True:
      for i, targets_set in enumerate(targets):
        for j, targets_person in enumerate(targets_set):
          targets_onehot = []
          for k in range(targets_person.shape[0]):
            targets_onehot.append(label_dict[targets_person[k][0]])
          dataset.targets[i][j] = np.array(targets_onehot)
    else:
      targets_onehot = []
      for i in range(targets.shape[0]):
        targets_onehot.append(label_dict[targets[i][0]])
      dataset.targets = np.array(targets_onehot)

    train_set, val_set, test_set = dataset.partition()
    console.show_status('Finishing split dataset to (train_set, val_set, test_set)...')



    return train_set, val_set, test_set




  @classmethod
  def load_as_tframe_data(cls, data_dir, data_name = None, **kwargs) -> SleepSet:
    """Return: an instance of SleepSet whose `properties` attribute contains
       {'signal_groups': [<a list of SignalGroup>]}

      e.g., data_set = SleepEDFx(name=f'Sleep-EDF-Expanded{suffix_k}',
                                 signal_groups=signal_groups)
    """
    if data_name == 'sleepedf':
      from xsleep.slp_datasets.sleepedfx import SleepEDFx as DataSet
    elif data_name == 'ucddb':
      from xsleep.slp_datasets.ucddb import UCDDB as DataSet
    else: raise KeyError(f'!! Unknown dataset `{data_name}`')

    data_set = DataSet.load_as_tframe_data(data_dir, data_name, **kwargs)
    return data_set


  @classmethod
  def _get_tfd_file_path(cls, data_dir, data_name, **kwargs):
    suffix = kwargs['suffix']
    return os.path.join(data_dir, data_name, f'{data_name}{suffix}.tfds')



if __name__ == '__main__':
  from xslp_core import th
  th.data_config = 'sleepedf:0,1,2'
  dataset = SLPAgent.load_as_tframe_data(th.data_dir, suffix='-alpha')
  print()

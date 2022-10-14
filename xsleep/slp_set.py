import os.path
from typing import List
from tframe.data.sequences.seq_set import SequenceSet
from pictor.objects.signals.signal_group import DigitalSignal, SignalGroup

import numpy as np



class SleepSet(SequenceSet):

  STAGE_KEY = 'STAGE'

  # region: Properties

  @property
  def signal_groups(self) -> List[SignalGroup]:
    return self.properties['signal_groups']

  # endregion: Properties

  # region: APIs

  @classmethod
  def load_as_tframe_data(cls, data_dir):
    raise NotImplementedError

  @classmethod
  def load_raw_data(cls, data_dir):
    raise NotImplementedError

  def configure(self,channel_select: str):
    """
    channel_select examples: '0,2,6'
    """
    raise NotImplementedError

  def report(self):
    raise NotImplementedError

  # endregion: APIs

  # region: Data IO

  @classmethod
  def read_edf_data_pyedflib(cls, fn: str, channel_list: List[str] = None,
                             freq_modifier=None, length = None) -> List[DigitalSignal]:
    """Read .edf file using pyedflib package.

    :param fn: file name
    :param channel_list: list of channels. None by default.
    :param freq_modifier: This arg is for datasets such as Sleep-EDF, in which
                          frequency provided is incorrect.
    :return: a list of DigitalSignals
    """
    import pyedflib

    # Sanity check
    assert os.path.exists(fn)

    signal_dict = {}
    with pyedflib.EdfReader(fn) as file:
      # Check channels
      all_channels = file.getSignalLabels()
      if channel_list is None: channel_list = all_channels
      # Read channels
      for channel_name in channel_list:
        # Get channel id
        chn = all_channels.index(channel_name)
        frequency = file.getSampleFrequency(chn)

        # Apply freq_modifier if provided
        if callable(freq_modifier): frequency = freq_modifier(frequency)

        # Initialize an item in signal_dict if necessary
        if frequency not in signal_dict: signal_dict[frequency] = []
        # Read signal
        signal_dict[frequency].append((channel_name, file.readSignal(chn)[:length]))

    # Wrap data into DigitalSignals
    digital_signals = []
    for frequency, signal_list in signal_dict.items():
      ticks = np.arange(len(signal_list[0][1])) / frequency
      ticks = ticks[:length]
      digital_signals.append(DigitalSignal(
        np.stack([x for _, x in signal_list], axis=-1), ticks=ticks,
        channel_names=[name for name, _ in signal_list],
        label=f'Freq=' f'{frequency}'))

    return digital_signals

  @classmethod
  def read_edf_anno_mne(cls, fn: str, allow_rename=True)-> list:
    from mne import read_annotations

    # Check extension
    if fn[-3:] != 'edf':
      if not allow_rename:
        # Rename .rec file if necessary, since mne package works only for
        # files with .rec extension
        raise TypeError(f'!! extension of `{fn}` is not .edf')
      os.rename(fn, fn + '.edf')
      fn = fn + '.edf'

    assert os.path.exists(fn)

    stage_anno = []
    raw_anno = read_annotations(fn)
    anno = raw_anno.to_data_frame().values
    anno_dura = anno[:, 1]
    anno_desc = anno[:, 2]
    for dura_num in range(len(anno_dura) - 1):
      for stage_num in range(int(anno_dura[dura_num]) // 30):
        stage_anno.append(anno_desc[dura_num])
    return stage_anno

  @classmethod
  def read_edf_data_mne(cls, fn: str, channel_list: List[str],
                        allow_rename=True) -> np.ndarray:
    """Read .edf file using `mne` package"""
    from mne.io import read_raw_edf
    from mne.io.edf.edf import RawEDF

    # Check extension
    if fn[-3:] != 'edf':
      if not allow_rename:
        # Rename .rec file if necessary, since mne package works only for
        # files with .rec extension
        raise TypeError(f'!! extension of `{fn}` is not .edf')
      os.rename(fn, fn + '.edf')
      fn = fn + '.edf'

    assert os.path.exists(fn)

    with read_raw_edf(fn, preload=True) as raw_edf:
      assert isinstance(raw_edf, RawEDF)
      channel_list = list(channel_list)
      edf_data = raw_edf.pick_channels(channel_list).to_data_frame().values

    return edf_data

  # endregion: Data IO

  # region: Data Configuration

  def format_data(self):
    from xslp_core import th

    features = self.features
    targets = self.targets
    sample_length = th.random_sample_length
    if th.use_rnn:
      for i, sg_data in enumerate(features):
        len = sg_data.shape[0]
        data_reshape = sg_data.reshape(len // sample_length, sample_length)
        targets_reshape = targets[i].reshape(len // sample_length, 1)
        features[i] = data_reshape
        targets[i] = targets_reshape
      person_num = int(th.data_config.split(':')[1])
      train_person = int(person_num * 0.7)
      val_person = int(person_num * 0.1)
      train_set_features = features[:train_person]
      train_set_targets = targets[:train_person]
      val_set_features = features[train_person:train_person+val_person]
      val_set_targets = targets[train_person:train_person+val_person]
      test_set_features = features[train_person + val_person:]
      test_set_targets = targets[train_person + val_person:]
      self.features = [train_set_features, val_set_features, test_set_features]
      self.targets = [train_set_targets, val_set_targets, test_set_targets]
    else:
      for i, sg_data in enumerate(features):
        len, chn = sg_data.shape[0], sg_data.shape[1]
        data_reshape = sg_data.reshape(len // sample_length, sample_length, chn)
        targets_reshape = targets[i].reshape(len // sample_length, 1)
        features[i] = data_reshape
        targets[i] = targets_reshape
      self.features = np.vstack(features[:])
      self.targets = np.vstack(targets[:])
    # Set targets


  def partition(self):
    from xslp_core import th
    def split_and_return(over_classes=True, random=True):
      train_ratio = 7
      val_ratio = 1
      test_ratio = 2

      self.properties[self.NUM_CLASSES] = 5
      data_sets = self.split(train_ratio, val_ratio, test_ratio,
                            random=random,
                            over_classes=over_classes)

      from tframe import DataSet
      for ds in data_sets: ds.__class__ = DataSet

      return data_sets

    def get_sequence_data(features:List, targets:List):
      features_list = []
      targets_list = []
      for i, feature in enumerate(features):
        nums = feature.shape[0] // 5
        for j in range(nums):
          features_list.append(feature[j * 5:(j + 1) * 5])
          targets_list.append(targets[i][j * 5:(j + 1) * 5])
      data_set = SequenceSet(features=features_list,
                             targets=targets_list,
                             name='SleepData')
      assert isinstance(data_set, SequenceSet)
      return data_set

    if th.use_rnn:
      train_set = get_sequence_data(self.features[0], self.targets[0])
      val_set = get_sequence_data(self.features[1], self.targets[1])
      test_set = get_sequence_data(self.features[2], self.targets[2])
      return [train_set, val_set, test_set]
    else:
      return split_and_return(over_classes=True, random=True)
  # endregion: Data Configuration

  # region: Visualization


  # def partition_lll(self):
  #   """th.data_config examples:
  #      (1) `95,1,1,1,1`
  #
  #   return [(train_1, val_1, test_1), (train_2, val_2, test_2), ...]
  #   """
  #   from lll_core import th
  #
  #   def split_and_return(index, data_set, over_classes=True, random=True):
  #     from tframe.data.dataset import DataSet
  #     # assert isinstance(data_set, DataSet)
  #     # if index == 0:
  #     #   train_ratio = 8.3
  #     #   val_ratio = 1
  #     #   test_ratio = 0.7
  #     # else:
  #     train_ratio = 7
  #     val_ratio = 1
  #     test_ratio = 2
  #
  #     names = [f'Train-{index+1}', f'Val-{index+1}', f'Test-{index+1}']
  #     data_sets = data_set.split(train_ratio,val_ratio,test_ratio,
  #                                random=random,
  #                                over_classes=over_classes,
  #                                names=names)
  #     # Show data info
  #     # cls._show_data_sets_info(data_sets)
  #     return data_sets
  #
  #   self.configure('0,1,2')
  #   index = 0
  #   datasets = []
  #   #split sleepedfx to (p1, p2, ...)
  #   for order, num in enumerate(th.data_config.split(',')):
  #     features = (np.vstack(self.features[index:index+int(num)]))
  #     targets = (np.vstack(self.targets[index:index+int(num)]))
  #     datasets.append(DataSet(features=features, targets=targets,
  #                             name=f'dataset-{order}'))
  #     # datasets.append(SleepEDFx(features=features, targets=targets,
  #     #                           name=f'data{order}',
  #     #                           signal_groups=self.signal_groups[index:index+int(num)]))
  #     index = index + int(num)
  #   for ds in datasets:
  #     ds.properties[self.NUM_CLASSES] = 5
  #   # split px to (train, val, test)
  #   for index, dataset in enumerate(datasets):
  #     assert isinstance(dataset, DataSet)
  #     train_set, val_set, test_set = split_and_return(index, dataset)
  #     datasets[index] = (train_set, val_set, test_set)
  #
  #   return datasets

  def show(self, channels: List[str] = None, **kwargs):
    from pictor import Pictor
    from pictor.plotters import Monitor

    p = Pictor(title='SleepSet', figure_size=(8, 6))
    p.objects = self.signal_groups
    p.add_plotter(Monitor(**kwargs))
    p.show()

  # endregion: Visualization

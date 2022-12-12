import os.path
from typing import List
from tframe.data.sequences.seq_set import SequenceSet
from pictor.objects.signals.signal_group import DigitalSignal, SignalGroup
from roma import console

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

  def configure(self, **kwargs):
    """
    channel_select examples: '0,2,6'
    """
    raise NotImplementedError

  def report(self):
    raise NotImplementedError

  # endregion: APIs

  # region: Data IO

  @classmethod
  def read_sleepedf_data_pyedflib(cls,
                                  fn: str,
                                  channel_list: List[str] = None,
                                  freq_modifier=None,
                                  label_index=None) -> List[DigitalSignal]:
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
        # Select only the data with labels
        select_idx = np.intersect1d(np.arange(len(file.readSignal(chn))),
                                    label_index[frequency])
        # Read signal
        signal_dict[frequency].append(
          (channel_name, file.readSignal(chn)[select_idx]))

    # Wrap data into DigitalSignals
    digital_signals = []
    for frequency, signal_list in signal_dict.items():
      ticks = np.arange(len(signal_list[0][1])) / frequency
      digital_signals.append(DigitalSignal(
        np.stack([x for _, x in signal_list], axis=-1), ticks=ticks,
        channel_names=[name for name, _ in signal_list],
        label=f'Freq=' f'{frequency}'))

    return digital_signals

  @classmethod
  def read_sleepedf_anno_mne(cls, fn: str, allow_rename=True) -> tuple:
    from mne import read_annotations

    ann2label = {'Sleep stage W': 0, 'Sleep stage 1': 1, 'Sleep stage 2': 2,
                 'Sleep stage 3': 3, 'Sleep stage 4': 3, 'Sleep stage R': 4,
                 'Movement time': 5, 'Sleep stage ?': 5}
    # Check extension
    if fn[-3:] != 'edf':
      if not allow_rename:
        # Rename .rec file if necessary, since mne package works only for
        # files with .rec extension
        raise TypeError(f'!! extension of `{fn}` is not .edf')
      os.rename(fn, fn + '.edf')
      fn = fn + '.edf'

    assert os.path.exists(fn)

    labels = []
    labels_index = {}  #{100:[], 1:[]}
    labels_index_high = []
    labels_index_low = []
    raw_anno = read_annotations(fn)
    anno = raw_anno.to_data_frame().values
    anno_onset = anno[:, 0]
    anno_dura = anno[:, 1]
    anno_desc = anno[:, 2]
    for index, duration in enumerate(anno_dura):
      h, m, s = anno_onset[index].strftime("%H:%M:%S").split(':')
      onset_timestamp = int(h) * 3600 + int(m) * 60 + int(s)
      duration_epoch = int(duration / 30)
      label = ann2label[anno_desc[index]]
      # if label != 5:
      label_epochs = np.ones(duration_epoch, dtype=np.int) * label
      labels.extend(label_epochs)
      idx_high_frequency = int(onset_timestamp * 100) + np.arange(duration * 100, dtype=np.int)
      idx_low_frequency = int(onset_timestamp) + np.arange(duration, dtype=np.int)
      labels_index_high.extend(idx_high_frequency)
      labels_index_low.extend(idx_low_frequency)
    labels_index[100] = labels_index_high
    labels_index[1] = labels_index_low
    return labels, labels_index

  @classmethod
  def read_rrsh_data_mne(cls, fn: str, channel_list: List[str] = None,
                         start=None, end=None) -> List[DigitalSignal]:
    """Read .edf file using `mne` package"""
    from mne.io import read_raw_edf
    from mne.io.edf.edf import RawEDF

    assert os.path.exists(fn)

    signal_dict = {}
    with read_raw_edf(fn, preload=False) as raw_data:
      assert isinstance(raw_data, RawEDF)
      # resample data to 100Hz
      frequency = raw_data.info['sfreq']
      if frequency != 100:
        raw_data = raw_data.resample(100)
        frequency = raw_data.info['sfreq']
      # Check channels
      all_channels = raw_data.ch_names
      all_data = raw_data.get_data()
      if channel_list is None: channel_list = all_channels
      # Read Channels
      for channel_name in channel_list:
        chn = all_channels.index(channel_name)
        if frequency not in signal_dict: signal_dict[frequency] = []
        signal_dict[frequency].append(
          (channel_name, all_data[chn][start:end]))

    # Wrap data into DigitalSignals
    digital_signals = []
    for frequency, signal_list in signal_dict.items():
      ticks = np.arange(len(signal_list[0][1])) / frequency
      digital_signals.append(DigitalSignal(
        np.stack([x for _, x in signal_list], axis=-1), ticks=ticks,
        channel_names=[name for name, _ in signal_list],
        label=f'Freq=' f'{frequency}'))

    return digital_signals

  @classmethod
  def read_rrsh_anno_xml(cls, fn: str, allow_rename=True) -> list:
    import xml.dom.minidom as xml


    if fn[-3:] != 'XML':
      if not allow_rename:
        # Rename .rec file if necessary, since mne package works only for
        # files with .rec extension
        raise TypeError(f'!! extension of `{fn}` is not .edf')
      os.rename(fn, fn + '.XML')
      fn = fn + '.XML'
    assert os.path.exists(fn)

    dom = xml.parse(fn)
    root = dom.documentElement
    sleep_stages = root.getElementsByTagName('SleepStage')
    stage_anno = [int(stage.firstChild.data) for stage in sleep_stages]
    stage_anno = [4 if i == 5 else i for i in stage_anno]
    stage_anno = [5 if i == 9 else i for i in stage_anno]

    return stage_anno

  # endregion: Data IO

  # region: Data Configuration

  def format_data(self):
    console.show_status(f'Formating data...')
    features = self.features
    targets = self.targets

    for i, sg_features in enumerate(features):
      features_reshape = np.asarray(np.split(sg_features, len(targets[i])))
      features[i] = features_reshape

    # Set features
    self.features = features
    console.show_status(f'Finishing formating data...')

  def partition(self, train_ratio):
    from tframe import hub as th

    features = self.features
    targets = self.targets
    # split data to train_set and test_data
    if th.test_config:
      test_index = [int(i) for i in th.test_config.split(':')[1].split(',')]
    else: test_index = [0,1]
    train_index = np.setdiff1d(np.arange(int(th.data_config.split(':')[1])), test_index)
    test_feature = np.vstack(np.array(features, dtype=object)[test_index])
    test_label = np.vstack(np.array(targets, dtype=object)[test_index])
    train_feature = np.vstack(np.array(features, dtype=object)[train_index])
    train_label = np.vstack(np.array(targets, dtype=object)[train_index])
    from sleepedfx import SleepEDFx
    test_set = SleepEDFx(name=f'Sleep-EDF-Expanded',
                         features=test_feature,
                         targets=test_label)
    train_set = SleepEDFx(name=f'Sleep-EDF-Expanded',
                         features=train_feature,
                         targets=train_label)
    test_set.properties[self.NUM_CLASSES] = 5
    train_set.properties[self.NUM_CLASSES] = 5

    # Split channel if necessary
    if ';' in th.channels:
      for i, channels in enumerate(th.fusion_channels):
        test_set.data_dict[f'input-{i+1}'] = np.stack(
          [test_feature[:, :, int(c)] for c in channels], axis=-1
        )
        train_set.data_dict[f'input-{i+1}'] = np.stack(
          [train_feature[:, :, int(c)] for c in channels], axis=-1
        )

    # Split train_set to train_set and validate_set
    train_num = int(train_label.shape[0] * train_ratio)
    val_num = int(train_label.shape[0] - train_num)
    train_set, validate_set = train_set.split(train_num, val_num,
                                         random=True, over_classes=True)
    data_sets = [train_set, validate_set, test_set]
    from tframe import DataSet
    for ds in data_sets: ds.__class__ = DataSet

    return data_sets

  def get_sequence_data(self, features: List, targets: List):
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

  # endregion: Data Configuration

  # region: Visualization

  def show(self):
    from pictor import Pictor
    from pictor.plotters import Monitor

    p = Pictor(title='SleepSet', figure_size=(8, 6))
    p.objects = self.signal_groups
    p.add_plotter(Monitor())
    p.show()

  # endregion: Visualization



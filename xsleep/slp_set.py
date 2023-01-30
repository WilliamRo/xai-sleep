from pictor.objects.signals.digital_signal import DigitalSignal
from pictor.objects.signals.signal_group import SignalGroup, Annotation
from roma import io
from tframe.data.sequences.seq_set import SequenceSet, DataSet
from tframe import console

from typing import List
import numpy as np
import os.path


class SleepSet(SequenceSet):

  class Keys:
    tapes = 'SleepSet::Keys::tapes'

  ANNO_KEY = 'stage Ground-Truth'
  EPOCH_DURATION = 30.0

  CHANNELS = {}

  # region: Properties

  @property
  def signal_groups(self) -> List[SignalGroup]:
    return self.properties['signal_groups']

  # endregion: Properties

  # region: APIs

  @classmethod
  def load_as_sleep_set(cls, data_dir):
    raise NotImplementedError

  @classmethod
  def load_as_signal_groups(cls, data_dir):
    raise NotImplementedError

  def configure(self, **kwargs):
    """
    channel_select examples: '0,2,6'
    """
    raise NotImplementedError

  def report(self):
    raise NotImplementedError

  @staticmethod
  def save_sg_file_if_necessary(pid, sg_path, n_patients, i, sg, **kwargs):
    if kwargs.get('save_sg', True):
      console.show_status(f'Saving `{pid}` data ...')
      console.print_progress(i, n_patients)
      io.save_file(sg, sg_path)
      console.show_status(f'Data saved to `{sg_path}`.')

  # endregion: APIs

  # region: Data IO
  @staticmethod
  def read_digital_signals_mne(
          file_path: str,
          groups=None,
          dtype=np.float32,
          **kwargs
  ) -> List[DigitalSignal]:
    """Read .edf file using `mne` package.

    :param groups: A list/tuple of channel names groups by sampling frequency.
           If not provided, data will be read in a channel by channel fashion.
    :param max_sfreq: maximum sampling frequency
    :param allow_rename: option to allow rename file when target extension is
           not .edf.
    """
    import mne.io

    max_sfreq = kwargs.get('max_sfreq', None)

    # Rename file if necessary
    file_path = file_path
    if file_path[-4:] != '.edf' and kwargs.get('allow_rename', False):
      os.rename(file_path, file_path + '.edf')
      file_path += '.edf'

    open_file = lambda exclude=(): mne.io.read_raw_edf(
      file_path, exclude=exclude, preload=False, verbose=False)

    # Initialize groups if not provided, otherwise get channel_names from groups
    if groups is None:
      with open_file() as file:
        channel_names = file.ch_names
      groups = [[chn] for chn in channel_names]
    else:
      channel_names = [chn for g in groups for chn in g]

    # Generate exclude lists
    exclude_lists = [[chn for chn in channel_names if chn not in g]
                     for g in groups]

    # Read raw data {sfreq:[(ch_name, data),(),..,()], sfreq:[(),..], ..}
    signal_dict = {}
    for exclude_list in exclude_lists:
      with open_file(exclude_list) as file:
        sfreq = file.info['sfreq']

        # Resample to `max_sfreq` if necessary
        if max_sfreq is not None and sfreq > max_sfreq:
          file.resample(max_sfreq)
          sfreq = max_sfreq

        # Read signal
        if sfreq not in signal_dict: signal_dict[sfreq] = []
        signal_dict[sfreq].append((file.ch_names, file.get_data()))

    # Wrap data into DigitalSignals
    digital_signals = []
    for sfreq, signal_lists in signal_dict.items():
      data = np.concatenate([x for _, x in signal_lists], axis=0)
      data = np.transpose(data).astype(dtype)
      channel_names = [name for names, _ in signal_lists for name in names]
      digital_signals.append(DigitalSignal(
        data, channel_names=channel_names, sfreq=sfreq,
        label=','.join(channel_names)))

    return digital_signals

  @staticmethod
  def read_annotations_mne(file_path: str, labels=None) -> Annotation:
    """Read annotations using `mne` package"""
    import mne

    # Read mne.Annotations
    mne_anno: mne.Annotations = mne.read_annotations(file_path)

    # Automatically generate labels if necessary
    if labels is None: labels = list(sorted(set(mne_anno.description)))

    # Read intervals and annotations
    intervals, annotations = [], []
    label2int = {lb: i for i, lb in enumerate(labels)}
    for onset, duration, label in zip(
            mne_anno.onset, mne_anno.duration, mne_anno.description):
      intervals.append((onset, onset + duration))
      annotations.append(label2int[label])

    return Annotation(intervals, annotations, labels=labels)

  @staticmethod
  def read_rrsh_data_mne(fn: str, channel_list: List[str] = None,
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

  @staticmethod
  def read_rrsh_anno_xml(fn: str, allow_rename=True) -> List:
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

  @staticmethod
  def try_to_load_sg_directly(
          pid, sg_path, n_patients, i, signal_groups, **kwargs):
    console_symbol = f'[{i + 1}/{n_patients}]'
    if os.path.exists(sg_path) and not kwargs.get('overwrite', False):
      console.show_status(
        f'Loading `{pid}` data from `{sg_path}` ...', symbol=console_symbol)
      console.print_progress(i, n_patients)
      sg = io.load_file(sg_path)
      signal_groups.append(sg)
      return True

    # Otherwise, create sg from raw file
    console.show_status(f'Reading `{pid}` data ...', symbol=console_symbol)
    console.print_progress(i, n_patients)
    return False
  # endregion: Data IO

  # region: Data Configuration
  def remove_wake_signal(self, config='terry'):
    assert config == 'terry'
    # For each patient
    for sg in self.signal_groups:
      # cut annotations
      annotation = sg.annotations[self.ANNO_KEY]
      label = annotation.annotations
      wake_index = np.argwhere(np.array(label) == 0)
      wake_begin, wake_end = wake_index[0][0], wake_index[-1][0]
      intervals = annotation.intervals
      intervals[wake_begin] = (max(intervals[wake_begin][0], intervals[wake_begin][1]-1800), intervals[wake_begin][1])
      intervals[wake_end] = (intervals[wake_end][0], min(intervals[wake_end][0]+1800, intervals[wake_end][1]))

      for ds in sg.digital_signals:
        # TODO
        freq = int(ds.sfreq)
        _start, _end = intervals[wake_begin][0] * freq, intervals[wake_end][1] * freq
        ds.data = ds.data[int(_start):int(_end)]

  def interpolate(self, sg, chn_names) -> List:
    """
     convert channels to the same sampling frequency: 100hz
    -------------------------------------------------------------
    Parameters:
     :param sg: sigal person
     :param chn_names: channel you selected
     :return: chn_data
    """
    chn_data = []
    for name in chn_names:
      feq = float(sg.channel_signal_dict[name].sfreq)
      if feq == 1:
        from scipy import interpolate
        x_old = np.linspace(0, 1, len(sg[name]))
        y_old = sg[name]
        x = np.linspace(0, 1, len(x_old) * 100)
        f = interpolate.interp1d(x_old, y_old, kind='linear')
        y = f(x)
        sg[name] = y
        chn_data.append(y)
      else:
        chn_data.append(sg[name])
    return chn_data

  def add_noise(self, features, targets, chn_names, remove_data_total, select_label_index_total):
    from tframe import hub as th
    from random import sample

    for index, feature in enumerate(features):
      select_data_index = []
      target = targets[index]
      target_index = np.arange(len(target))
      bad_index = sample(list(target_index), int(len(target_index) * th.ratio))
      for i in bad_index:
        assert len(remove_data_total) > 0
        # seed = np.random.randint(0, len(remove_data_total), dtype=np.int)
        temp_index = i * 3000 + np.arange(3000, dtype=np.int)
        random_channel = [np.random.randint(0, len(chn_names), dtype=np.int) for j in range(len(chn_names) - 1)]
        random_channel = list(set(random_channel))
        for i in random_channel:
          # feature[list(temp_index), i] = remove_data_total[seed][:, i]
          feature[list(temp_index), i] = 0
      features[index] = feature

      if th.show_in_monitor:
        from scipy import signal
        sg = self.signal_groups[index]
        select_label_index = select_label_index_total[index]
        for i, name in enumerate(chn_names):
          freq = float(sg.channel_signal_dict[name].sfreq)
          for label_index in select_label_index:
            data_index = int(label_index * 30 * freq) + np.arange(30 * freq, dtype=np.int)
            select_data_index.extend(data_index)
          data = feature[:, i]
          if freq == 1:
            data = signal.resample(data, data.shape[0] // 100)
          sg[name][select_data_index] = data
          select_data_index.clear()
        self.signal_groups[index] = sg

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
      for i, channels in enumerate(self.fusion_channels(th.channels)):
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

  def fusion_channels(self, channels):
    return [s.split(',') for s in channels.split(';')]

  def show(self):
    from pictor import Pictor
    from pictor.plotters import Monitor

    p = Pictor(title='SleepSet', figure_size=(8, 6))
    p.objects = self.signal_groups
    p.add_plotter(Monitor())
    p.show()

  # endregion: Visualization



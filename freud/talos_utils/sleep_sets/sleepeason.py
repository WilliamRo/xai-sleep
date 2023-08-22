from fnmatch import fnmatch
from freud.talos_utils.slp_config import SleepConfig
from freud.talos_utils.slp_set import SleepSet, DataSet
from pictor.objects.signals.signal_group import SignalGroup, DigitalSignal
from pictor.objects.signals.signal_group import Annotation
from roma.spqr.finder import walk
from roma import io
from tframe import console
from typing import List

import numpy as np
import os
import re



class SleepEason(SleepSet):

  FILE_LIST_KEY = 'file_list'

  def __init__(self, data_dir=None, buffer_size=None, file_list=None,
               name='no-name'):
    """buffer_size decides how many files to fetch per round
    """
    self.buffer_size = buffer_size
    self.name = name
    self.data_dict = {}

    # Initialize properties
    self.properties = {'CLASSES': ['Wake', 'N1', 'N2', 'N3', 'REM']}

    # Set file list
    assert (data_dir is None and file_list is not None or
            data_dir is not None and file_list is None)
    self.properties[self.FILE_LIST_KEY] = (
      walk(data_dir, pattern='*.sg') if file_list is None else file_list)

    # Set data fetcher
    self.data_fetcher = self.fetch_data

    # Necessary fields to prevent errors
    self.is_rnn_input = False

  # region: Properties

  @property
  def size(self): return len(self.file_list)

  @property
  def file_list(self): return self.properties[self.FILE_LIST_KEY]

  @property
  def num_signal_groups(self) -> int: return len(self.file_list)

  @SleepSet.property()
  def validation_set(self) -> DataSet:
    shadow = self.get_subset_by_patient_id()
    shadow.buffer_size = None
    shadow._fetch_data()
    return shadow.extract_data_set(include_targets=True)

  # endregion: Properties

  # region: Public Methods

  @staticmethod
  def fetch_data(self):
    if self.buffer_size is None: files = self.file_list
    else: files = np.random.choice(
      self.file_list, self.buffer_size, replace=False)

    console.show_status(f'Fetching signal groups to {self.name} ...')

    # Release memory (TODO: CRUCIAL)
    if 'signal_groups' in self.properties:
      # TODO: without this line, model will be trained on same batch of file
      self._cloud_pocket.pop('epoch_table')

      for sg in self.signal_groups:
        assert isinstance(sg, SignalGroup)
        for ds in sg.digital_signals: ds.release()
        sg.release()

    self.signal_groups = []

    # Trigger garbage collection
    for p in files:
      sg = io.load_file(p)
      console.supplement(f'Loaded `{p}`', level=2)
      self.signal_groups.append(sg)

    # Extract tapes for each sg
    self.extract_sg_tapes()

  def _fetch_data(self):
    self.fetch_data(self)

  # endregion: Public Methods

  # region: Overwriting

  def configure(self):
    pass

  @classmethod
  def load_as_sleep_set(cls, data_dir, **kwargs):
    from tframe import hub as th
    return SleepEason(
      data_dir, buffer_size=th.sg_buffer_size, name='SleepEason')

  def get_subset_by_patient_id(self, indices=None, name_suffix=''):
    if name_suffix != '': name_suffix = '-' + name_suffix
    if indices is None: indices = list(range(self.num_signal_groups))
    return SleepEason(buffer_size=self.buffer_size,
                      name=f'{self.name}{name_suffix}',
                      file_list=[self.file_list[i] for i in indices])

  # endregion: Overwriting

  # region: Generator

  @classmethod
  def find_signals_by_name(cls, ds: DigitalSignal, rule='alpha',
                           lower_match_dict=None):
    assert rule == 'alpha'

    if lower_match_dict is None: lower_match_dict = {
      'EEG': ['eeg*', 'c3a2', 'c4a1', '[fco]?-m[12]'],
      'EOG': ['eog*', 'lefteye', 'righteye', 'e?-m2'],
      'EMG': ['emg*', 'chin 1-chin 2'],
      'ECG': ['*ecg*']
    }

    # Fill-in prefixes and pattern according to lower_match_dict
    prefixes, patterns = [], []
    for key, pats in lower_match_dict.items():
      for p in pats:
        prefixes.append(key)
        patterns.append(p)

    channel_names, data_list, offset = [], [], None
    for src_name in ds.channels_names:
      matched_keys = [k for k, p in zip(prefixes, patterns)
                      if fnmatch(src_name.lower(), p)]
      if len(matched_keys) == 0: continue

      # Append data
      data_list.append(ds[src_name])

      # Append channel name
      prefix = matched_keys[0]
      tgt_name = src_name
      if not tgt_name.startswith(prefix): tgt_name = f'{prefix} {tgt_name}'
      channel_names.append(tgt_name)

      # Set offset
      offset = ds.off_set

    return channel_names, data_list, offset

  @classmethod
  def convert_to_eason_sg(cls, src_dir, tgt_dir, src_pattern='*.sg',
                          format='alpha',  file_prefix=''):
    """Format details:

    alpha
    -----
    1. Contains EEG, EOG, EMG, ECG channels, fs=128Hz;
    2. preprocess=IQR
    """
    assert format == 'alpha'
    sfreq = 128

    # Read sg files from src_dir
    sg_file_list = walk(src_dir, 'file', src_pattern, return_basename=True)

    # Process and save each sg file to tgt_dir
    N = len(sg_file_list)
    for i, sg_fn in enumerate(sg_file_list):
      sg_path = os.path.join(src_dir, sg_fn)

      # Show status
      console_symbol = f'[{i + 1}/{N}]'
      console.show_status(
        f'Loaded `{sg_fn}` data from `{src_dir}`.', symbol=console_symbol)

      src_sg: SignalGroup = io.load_file(sg_path)

      # Extract digital signal
      data_list, channel_names, offset = [], [], None
      for src_ds in src_sg.digital_signals:
        if src_ds.sfreq != sfreq: continue

        # Find channels
        names, data, offset = cls.find_signals_by_name(src_ds)
        channel_names.extend(names)
        data_list.extend(data)

      # Continue if found no valid signal
      if len(data_list) == 0:
        console.show_status(f'!! failed to find valid signal in `{sg_fn}`')
        continue

      data = np.stack(data_list, axis=-1)
      ds = DigitalSignal(data, sfreq=sfreq, channel_names=channel_names,
                         off_set=offset, label=','.join(channel_names))
      tgt_sg = SignalGroup(ds, label=src_sg.label, **src_sg.properties)
      tgt_sg.label = file_prefix + tgt_sg.label
      tgt_sg.annotations = src_sg.annotations

      # Save sg file to target_dir
      tgt_sg_fn = file_prefix + re.sub('\([\d\w;,-]*\)', '', sg_fn)
      io.save_file(tgt_sg, os.path.join(tgt_dir, tgt_sg_fn))
      console.show_status(
        f'`{tgt_sg_fn}` data extracted and saved to `{tgt_dir}`.',
        symbol=console_symbol)

      # Print progress
      console.print_progress(i, N)

    console.show_status(f'Successfully converted {N} files.')

  # endregion: Generator

  # region: Benchmark SG indices

  BENCHMARK = {
    'alpha': {'val': ['SC4312', 'SC4422', 'ucddb025', 'ucddb026',
                      'rrsh-ZJK', 'rrsh-ZGC'],
              'test': ['SC4482', 'SC4651', 'ucddb027', 'ucddb028',
                       'rrsh-ZYJ', 'rrsh-ZSQ']},
  }

  def split(self):
    """Split self to train/val/test datasets.
    Example th.data_config syntax: `sleepeason1 EEGx2,EOGx1 alpha`
    """
    from tframe import hub as th
    from freud.talos_utils.slp_config import SleepConfig

    assert isinstance(th, SleepConfig)
    key = th.data_args[1]
    val_keys, test_keys = [self.BENCHMARK[key][k] for k in ('val', 'test')]

    # Filter file_list if required
    file_list = self.file_list
    if 'pattern' in th.data_kwargs:
      p = th.data_kwargs['pattern']
      file_list = [s for s in file_list if re.match(p, s.lower()) is not None]

    train_file_list, val_file_list, test_file_list = [], [], []
    # Construct file_lists
    for fn in file_list:
      flag = False
      for k in val_keys:
        if k in fn:
          val_file_list.append(fn)
          flag = True
          break
      if flag: continue
      for k in test_keys:
        if k in fn:
          test_file_list.append(fn)
          flag = True
          break
      if flag: continue
      train_file_list.append(fn)

    # Create datasets
    train_set = SleepEason(name='TrainSet', file_list=train_file_list,
                           buffer_size=th.sg_buffer_size)
    val_set = SleepEason(name='ValSet', file_list=val_file_list)
    test_set = SleepEason(name='TestSet', file_list=test_file_list)

    assert train_set.size + val_set.size + test_set.size == len(file_list)

    return [train_set, val_set, test_set]

  # endregion: Benchmark SG indices



if __name__ == '__main__':
  from pprint import pprint

  data_dir = r'../../../data/sleepeason1'

  se = SleepEason(data_dir, buffer_size=10)
  se.fetch_data()

  # pprint(se.properties)

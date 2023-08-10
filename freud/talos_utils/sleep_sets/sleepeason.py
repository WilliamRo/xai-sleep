from freud.talos_utils.slp_config import SleepConfig
from freud.talos_utils.slp_set import SleepSet, DataSet
from pictor.objects.signals.signal_group import SignalGroup, DigitalSignal
from pictor.objects.signals.signal_group import Annotation
from roma.spqr.finder import walk
from roma import io
from tframe import console
from typing import List

import os
import numpy as np



class SleepEason(SleepSet):

  FILE_LIST_KEY = 'file_list'

  # TODO: this is necessary for extracting tapes, needed to be refactored
  CHANNELS = {'1': 'EEG Fpz-Cz',
              '2': 'EEG Pz-Oz',
              '3': 'EOG horizontal',
              '4': 'Resp oro-nasal',
              '5': 'EMG submental',
              '6': 'Temp rectal',
              '7': 'Event marker'}

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
    self.signal_groups = []
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
  def convert_to_eason_sg(cls, src_dir, tgt_dir, src_pattern=None, format='alpha'):
    from roma.spqr.finder import walk

    """Format details:

    alpha
    -----
    1. EEG, EOG, EMG fs=128Hz;
    """
    assert format == 'alpha'
    if src_pattern== None: src_pattern = '*(trim1800;iqr,1,20;128).sg'
    sg_file_list = walk(src_dir, 'file', src_pattern,
                        return_basename=True)
    n_patients = len(sg_file_list)
    origin_signalgroups = []

    for i, sg_file in enumerate(sg_file_list):
      pid = sg_file[:7]
      sg_path = os.path.join(src_dir, sg_file)
      cls.try_to_load_sg_directly(pid, sg_path, n_patients, i, origin_signalgroups)
      sg = origin_signalgroups[i]

      for ds in sg.digital_signals:
        names = []
        eeg_sign = ['EEG', 'F3-M2', 'C3', 'O1', 'F4', 'C4-M1', 'O2-M1', 'C3', 'C4']
        eog_sign = ['EOG', 'eye', 'Eye', 'E1-', 'E2-']
        emg_sign = ['Chin', 'EMG']
        ecg_sign = ['ECG']
        for name in ds.channels_names:
          if any(sign in name for sign in eeg_sign): name = 'EEG:' + name
          if any(sign in name for sign in eog_sign): name = 'EOG:' + name
          if any(sign in name for sign in emg_sign): name = 'EMG:' + name
          if any(sign in name for sign in ecg_sign): name = 'ECG:' + name

          names.append(name)

        ds.channels_names = names

      sg_save_path = os.path.join(tgt_dir, sg_file)
      cls.save_sg_file_if_necessary(
        pid, sg_save_path, n_patients, i, sg, save_sg=True)

  # endregion: Generator



if __name__ == '__main__':
  from pprint import pprint

  data_dir = r'../../../data/sleepeason1'

  se = SleepEason(data_dir, buffer_size=10)
  se.fetch_data()

  # pprint(se.properties)

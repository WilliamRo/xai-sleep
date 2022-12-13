import os

from pictor.objects.signals.signal_group import DigitalSignal, SignalGroup
from tframe.data.sequences.seq_set import SequenceSet
from typing import List, Optional, Union, Callable

import numpy as np



class SleepSet(SequenceSet):

  # region: Properties

  @property
  def signal_groups(self) -> List[SignalGroup]:
    return self.properties['signal_groups']

  # endregion: Properties

  # region: Overwriting

  def _check_data(self): pass

  # endregion: Overwriting

  # region: Abstract Methods

  @classmethod
  def load_as_signal_groups(cls, data_dir, **kwargs) -> List[SignalGroup]:
    raise NotImplementedError

  @classmethod
  def load_as_sleep_set(cls, data_dir, **kwargs) -> SequenceSet:
    raise NotImplementedError

  # endregion: Abstract Methods

  # region: Data Reading

  @staticmethod
  def read_digital_signals_mne(
      file_path: str,
      groups=None,
      dtype=np.float32,
  ) -> List[DigitalSignal]:
    """Read .edf file using `mne` package.

    :param groups: A list/tuple of channel names groups by sampling frequency.
           If not provided, data will be read in a channel by channel fashion.
    """
    import mne.io

    open_file = lambda exclude=(): mne.io.read_raw_edf(
      file_path, exclude=exclude, preload=False, verbose=False)

    # Initialize groups if not provided, otherwise get channel_names from groups
    if groups is None:
      with open_file() as file: channel_names = file.ch_names
      groups = [[chn] for chn in channel_names]
    else:
      channel_names = [chn for g in groups for chn in g]

    # Generate exclude lists
    exclude_lists = [[chn for chn in channel_names if chn not in g]
                     for g in groups]

    # Read raw data
    signal_dict = {}
    for exclude_list in exclude_lists:
      with open_file(exclude_list) as file:
        sfreq = file.info['sfreq']
        if sfreq not in signal_dict: signal_dict[sfreq] = []

        # Read signal
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

  # endregion: Data Reading



if __name__ == '__main__':
  data_root = r'E:\xai-sleep\data'

  edf_path = [
    r'sleepedf\SC4001E0-PSG.edf',
    r'ucddb\ucddb002.rec.edf',
    r'rrsh\JJF.edf',
  ][2]

  file_path = os.path.join(data_root, edf_path)

  for ds in SleepSet.read_digital_signals_mne(file_path): print(ds)

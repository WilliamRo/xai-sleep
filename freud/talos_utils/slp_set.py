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
      dtype=np.float32,
      groups=None,
  ) -> List[DigitalSignal]:
    """Read .edf file using `mne` package.

    :param groups: A list/tuple of channel names groups by sampling frequency.
           If not provided, data will be read in a channel by channel fashion.
    """
    import mne.io


    
    return []


  @staticmethod
  def read_digital_signals(
      file_path: str,
      exclude=(),
      freq_spec: Optional[Union[dict, float, Callable]] = None,
      use_package='mne',
      dtype=np.float32,
  ) -> List[DigitalSignal]:
    """Read .edf file using `pyedflib` or `mne` package.

    :param exclude: channels to exclude;
    :param freq_spec: frequency specification. If provided, it should be
           (1) a dictionary of '<chn_name>': <freq>, or
           (2) a float number specifying global frequency, or
           (3) a callable function with signature f(chn_name, freq).
           This parameter should not be set unless necessary.
    """
    # Find corresponding methods for each package
    if use_package == 'pyedflib':
      import pyedflib
      open_file = lambda fp: pyedflib.EdfReader(fp)
      get_sfreq = lambda file, chn: file.getSampleFrequency(chn)
      get_all_channels = lambda file: file.getSignalLabels()
      read_signal = lambda file, chn: file.readSignal(chn)
    elif use_package == 'mne':
      import mne.io
      open_file = lambda fp: mne.io.read_raw_edf(
        fp, exclude=exclude, preload=False, verbose=False)
      get_sfreq = lambda file, chn: file.info['sfreq']
      get_all_channels = lambda file: file.ch_names
      read_signal = lambda file, chn: file.get_data(chn).ravel()
    else: raise KeyError(f'!! Unknown package `{use_package}`')

    # Read raw data
    signal_dict = {}
    with open_file(file_path) as file:
      # Find channels to load
      all_channels = get_all_channels(file)

      # Read channels
      for channel_name in [cn for cn in all_channels if cn not in exclude]:
        # Get channel id
        chn = all_channels.index(channel_name)

        # Find sampling frequency
        frequency = get_sfreq(file, chn)
        if isinstance(freq_spec, float):
          frequency = freq_spec
        elif isinstance(freq_spec, dict) and channel_name in freq_spec:
          frequency = freq_spec[channel_name]
        elif callable(freq_spec):
          frequency = freq_spec(channel_name, frequency)
        else: assert freq_spec is None

        # Initialize an item in signal_dict if necessary
        if frequency not in signal_dict: signal_dict[frequency] = []

        # Read signal
        signal_dict[frequency].append((channel_name, read_signal(file, chn)))

    # Wrap data into DigitalSignals
    digital_signals = []
    for frequency, signal_list in signal_dict.items():
      data = np.stack([x for _, x in signal_list], axis=-1).astype(dtype)
      channel_names = [name for name, _ in signal_list]
      digital_signals.append(DigitalSignal(
        data, channel_names=channel_names,
        sfreq=frequency, label=','.join(channel_names)))

    return digital_signals

  # endregion: Data Reading



if __name__ == '__main__':
  from roma import console

  data_root = r'E:\xai-sleep\data'

  edf_path = [
    r'sleepedf\SC4001E0-PSG.edf',
    r'ucddb\ucddb002.rec.edf',
    r'rrsh\JJF.edf',
  ][0]

  pkg = ['pyedflib', 'mne'][0]

  file_path = os.path.join(data_root, edf_path)

  exclude = [['EEG Fpz-Cz',
              'EEG Pz-Oz',
              'EOG horizontal'],
             ['Resp oro-nasal',
              'EMG submental',
              'Temp rectal',
              'Event marker']][0]

  for ds in SleepSet.read_digital_signals(file_path, use_package=pkg,
                                          exclude=exclude):
    print(ds)

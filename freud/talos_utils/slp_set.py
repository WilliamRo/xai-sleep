from pictor.objects.signals.digital_signal import DigitalSignal
from pictor.objects.signals.signal_group import SignalGroup, Annotation
from tframe.data.sequences.seq_set import SequenceSet
from typing import List, Optional, Union, Callable
from tframe import console
from roma import io

import numpy as np
import os



class SleepSet(SequenceSet):

  ANNO_KEY = 'stage Ground-Truth'

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
    sg = cls.load_as_signal_groups(data_dir, **kwargs)
    return cls(name=str(cls.__class__), signal_groups=sg)

  # endregion: Abstract Methods

  # region: Data Reading

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
    if file_path[-4:] != '.edf' and kwargs.get('allow_rename', False):
      os.rename(file_path, file_path + '.edf')
      file_path += '.edf'

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

  # region: Common Utilities

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

  @staticmethod
  def save_sg_file_if_necessary(pid, sg_path, n_patients, i, sg, **kwargs):
    if kwargs.get('save_sg', True):
      console.show_status(f'Saving `{pid}` data ...')
      console.print_progress(i, n_patients)
      io.save_file(sg, sg_path)
      console.show_status(f'Data saved to `{sg_path}`.')

  # endregion: Common Utilities

  # endregion: Data Reading

  # region: Visualization

  def show(self, *funcs, **kwargs):
    from freud.gui.freud_gui import Freud

    # Initialize pictor and set objects
    freud = Freud(title=str(self.__class__.__name__))
    freud.objects = self.signal_groups

    for func in [func for func in funcs if callable(func)]: func(freud.monitor)

    for k, v in kwargs.items(): freud.monitor.set(k, v, auto_refresh=False)
    freud.show()

  # endregion: Visualization



if __name__ == '__main__':
  data_root = r'E:\xai-sleep\data'

  edf_path = [
    r'sleepedf\SC4001E0-PSG.edf',
    r'ucddb\ucddb002.rec.edf',
    r'rrsh\JJF.edf',
  ][2]

  file_path = os.path.join(data_root, edf_path)

  for ds in SleepSet.read_digital_signals_mne(file_path): print(ds)

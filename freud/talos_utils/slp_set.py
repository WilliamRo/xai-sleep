from freud.talos_utils.slp_config import SleepConfig
from pictor.objects.signals.digital_signal import DigitalSignal
from pictor.objects.signals.signal_group import SignalGroup, Annotation
from roma import io
from tframe.data.sequences.seq_set import SequenceSet, DataSet
from tframe import console
from typing import List

import numpy as np
import os



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

  @SequenceSet.property()
  def validation_set(self) -> DataSet:
    from tframe import hub as th
    assert isinstance(th, SleepConfig)

    if th.use_rnn: raise NotImplementedError(
      '!! RNN inputs are not supported currently')

    # Generate FNN inputs
    branches = [[] for _ in th.fusion_channels]
    for sg in self.signal_groups:
      tapes = sg.get_from_pocket(self.Keys.tapes)
      for branch, tape in zip(branches, tapes):
        branch.append(None)


    x, y = None, None
    return DataSet(x, y, name=f'{self.name}-val')

  # endregion: Properties

  # region: Overwriting

  def _check_data(self): pass

  # endregion: Overwriting

  # region: Methods for configuration

  def configure(self):
    """Configure dataset"""
    from tframe import hub as th
    assert isinstance(th, SleepConfig)

    # (1) extract required channels as tapes according to channel selection
    for sg in self.signal_groups:
      tapes = []
      for chn_lst in th.fusion_channels:
        tape = np.stack([sg[self.CHANNELS[key]] for key in chn_lst], axis=-1)
        tapes.append(tape)
      sg.put_into_pocket(self.Keys.tapes, tapes)

  # endregion: Methods for configuration

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
    from freud import read_digital_signals_mne
    return read_digital_signals_mne(file_path, groups, dtype, **kwargs)

  @staticmethod
  def read_annotations_mne(file_path: str, labels=None) -> Annotation:
    """Read annotations using `mne` package"""
    from freud import read_annotations_mne
    return read_annotations_mne(file_path, labels)

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
  ][0]

  file_path = os.path.join(data_root, edf_path)

  for ds in SleepSet.read_digital_signals_mne(file_path): print(ds)

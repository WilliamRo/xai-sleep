from freud.talos_utils.slp_set import SleepSet
from pictor.objects.signals.signal_group import DigitalSignal, SignalGroup
from roma.spqr.finder import walk
from tframe import console
from typing import List

import os
import numpy as np



class RRSHSC(SleepSet):
  """
  """

  CHANNEL_NAMES = []

  ANNO_LABELS = []

  # region: Data Loading

  @classmethod
  def load_as_signal_groups(cls, data_dir, **kwargs) -> List[SignalGroup]:
    """Directory structure of XXXXXX dataset is as follows:

       xxxxxxxxxx
         |- sleep-cassette
            |- SC4001E0-PSG.edf
            |- SC4001EC-Hypnogram.edf
            |- ...
         |- sleep-telemetry
            |- ST7011J0-PSG.edf
            |- ST7011JP-Hypnogram.edf
            |- ...


    Parameters
    ----------
    :param data_dir - a directory contains pairs of *-PSG.edf and
                      *-Hypnogram.edf
    """

    signal_groups: List[SignalGroup] = []

    # Traverse all hypnogram files
    hypno_file_names: List[str] = walk(data_dir, 'file', '*Hypnogram*',
                                       return_basename=True)
    n_patients = len(hypno_file_names)
    for i, hypno_fn in enumerate(hypno_file_names):
      # Parse patient ID and get find PSG file name
      pid = hypno_fn.split('-')[0][:7]

      # If the corresponding .sg file exists, read it directly
      sg_path = os.path.join(data_dir, pid + '(raw)' + '.sg')
      if cls.try_to_load_sg_directly(pid, sg_path, n_patients, i,
                                      signal_groups, **kwargs): continue

      # Create sg from raw file
      psg_fn = f'{pid}0-PSG.edf'

      # (1) read psg data as digital signals
      digital_signals: List[DigitalSignal] = cls.read_digital_signals_mne(
        os.path.join(data_dir, psg_fn), dtype=np.float16)

      # (2) read annotation
      annotation = cls.read_annotations_mne(
        os.path.join(data_dir, hypno_fn), labels=cls.ANNO_LABELS)

      # Wrap data into signal group
      sg = SignalGroup(digital_signals, label=f'{pid}')
      sg.annotations[cls.ANNO_KEY] = annotation
      signal_groups.append(sg)

      # Save sg if necessary
      cls.save_sg_file_if_necessary(pid, sg_path, n_patients, i, sg, **kwargs)

    console.show_status(f'Successfully read {n_patients} files.')
    return signal_groups

  # endregion: Data Loading



if __name__ == '__main__':
  import time

  console.suppress_logging()
  data_dir = r'../../../data/sleepedf'

  tic = time.time()
  ds = SleepEDFx.load_as_sleep_set(data_dir, overwrite=1)

  elapsed = time.time() - tic
  console.show_info(f'Time elapsed = {elapsed:.2f} sec.')

  # ds.show()



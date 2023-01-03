from freud.talos_utils.slp_set import SleepSet
from pictor.objects.signals.signal_group import DigitalSignal, SignalGroup
from roma.spqr.finder import walk
from tframe import console
from typing import List

import os
import numpy as np



class SleepEDFx(SleepSet):
  """The sleep-edf database contains 197 whole-night PolySomnoGraphic sleep
  recordings, containing EEG, EOG, chin EMG, and event markers. Some records
  also contain respiration and body temperature. Corresponding hypnograms
  (sleep patterns) were manually scored by well-trained technicians according
  to the Rechtschaffen and Kales manual, and are also available.
   The data comes from two studies, ...

  Reference: https://www.physionet.org/content/sleep-edfx/1.0.0/
  """

  CHANNEL_NAMES = ['EEG Fpz-Cz', 'EEG Pz-Oz', 'EOG horizontal',
                   'Resp oro-nasal', 'EMG submental', 'Temp rectal',
                   'Event marker']

  ANNO_LABELS = ['Sleep stage W', 'Sleep stage 1', 'Sleep stage 2',
                 'Sleep stage 3', 'Sleep stage 4', 'Sleep stage R',
                 'Movement time', 'Sleep stage ?']

  # region: Data Loading

  @classmethod
  def load_as_signal_groups(cls, data_dir, **kwargs) -> List[SignalGroup]:
    """Directory structure of SleepEDFx dataset is as follows:

       sleep-edf-database-expanded-1.0.0
         |- sleep-cassette
            |- SC4001E0-PSG.edf
            |- SC4001EC-Hypnogram.edf
            |- ...
         |- sleep-telemetry
            |- ST7011J0-PSG.edf
            |- ST7011JP-Hypnogram.edf
            |- ...

    However, this method supports loading SleepEDFx data from arbitrary
    folder, given that this folder contains SleepEDFx data.

    Parameters
    ----------
    :param data_dir: a directory contains pairs of *-PSG.edf and *-Hypnogram.edf
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
  data_dir = r'../../../data/sleepedfx'

  tic = time.time()
  ds = SleepEDFx.load_as_sleep_set(data_dir, overwrite=0)

  elapsed = time.time() - tic
  console.show_info(f'Time elapsed = {elapsed:.2f} sec.')

  ds.show()



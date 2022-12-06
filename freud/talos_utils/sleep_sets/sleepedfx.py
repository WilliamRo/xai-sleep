from freud.talos_utils.slp_set import SleepSet
from pictor.objects.signals.signal_group import DigitalSignal, SignalGroup
from roma.spqr.finder import walk
from tframe import console
from typing import List

import os



class SleepEDFx(SleepSet):
  """The sleep-edf database contains 197 whole-night PolySomnoGraphic sleep
  recordings, containing EEG, EOG, chin EMG, and event markers. Some records
  also contain respiration and body temperature. Corresponding hypnograms
  (sleep patterns) were manually scored by well-trained technicians according
  to the Rechtschaffen and Kales manual, and are also available.
   The data comes from two studies, ...

  Reference: https://www.physionet.org/content/sleep-edfx/1.0.0/
  """

  # region: Data Loading

  @classmethod
  def load_as_signal_groups(cls, data_dir, **kwargs) -> List[SignalGroup]:
    """Directory structure of raw dataset:

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
      id = hypno_fn.split('-')[0][:7]
      psg_fn = f'{id}0-PSG.edf'

      console.show_status(f'Reading `{id}` data ...',
                          symbol=f'[{i+1}/{n_patients}]')
      console.print_progress(i, n_patients)


    console.show_status(f'Successfully read {n_patients} files.')
    return signal_groups

  # endregion: Data Loading



if __name__ == '__main__':
  console.suppress_logging()
  data_dir = r'../../../data/sleepedf'

  signal_groups = SleepEDFx.load_as_signal_groups(data_dir)



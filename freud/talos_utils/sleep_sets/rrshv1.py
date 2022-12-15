from freud.talos_utils.slp_set import SleepSet
from pictor.objects.signals.signal_group import DigitalSignal, SignalGroup
from roma.spqr.finder import walk
from tframe import console
from typing import List

import os
import numpy as np



class RRSHSCv1(SleepSet):
  """This class is for wrapping data exported from Compumedics PSG devices.
  """

  CHANNEL_NAMES = ['E1-M2', 'E2-M2', 'Chin 1-Chin 2', 'F3-M2', 'C3-M2', 'O1-M2',
                   'F4-M1', 'C4-M1', 'O2-M1', 'RIP ECG', 'Pleth',
                   'Nasal Pressure', 'Therm', 'Thor', 'Abdo', 'Sum', 'SpO2',
                   'Snore', 'Leg/L', 'Leg/R', 'PositionSen', 'Pulse']

  ANNO_LABELS = ['Wake', 'N1', 'N2', 'N3', 'REM', 'Unknown']

  # region: Data Loading

  @classmethod
  def load_as_signal_groups(cls, data_dir, **kwargs) -> List[SignalGroup]:
    """Directory structure of RRSHSCv1 dataset is as follows:

       data-root
         |- CYG.edf                # PSG data
         |- CYG.xml                # annotation
         |- ...

    Parameters
    ----------
    :param data_dir: a directory contains pairs of *.edf and *.xml.XML files
    :param max_sfreq: maximum sampling frequency
    """
    import xml.dom.minidom as minidom

    max_sfreq = kwargs.get('max_sfreq', 128.)

    signal_groups: List[SignalGroup] = []

    # Traverse all .edf files
    edf_file_names: List[str] = walk(data_dir, 'file', '*.edf',
                                     return_basename=True)
    n_patients = len(edf_file_names)

    for i, edf_fn in enumerate(edf_file_names):
      # Parse patient ID and get find PSG file name
      pid = edf_fn.split('.')[0]

      # If the corresponding .sg file exists, read it directly
      sg_path = os.path.join(data_dir, pid + f'(max_sf_{max_sfreq}).sg')
      if cls.try_to_load_sg_directly(pid, sg_path, n_patients, i,
                                      signal_groups, **kwargs): continue

      # (1) read psg data as digital signals
      digital_signals: List[DigitalSignal] = cls.read_digital_signals_mne(
        os.path.join(data_dir, edf_fn), dtype=np.float16, max_sfreq=max_sfreq)

      # (2) read annotation
      xml_fp = os.path.join(data_dir, f'{pid}.xml')
      xml_root = minidom.parse(xml_fp).documentElement
      stage_elements = xml_root.getElementsByTagName('SleepStage')
      stages = np.array([int(se.firstChild.data) for se in stage_elements])
      stages[stages == 5] = 4
      stages[stages == 9] = 5

      # Wrap data into signal group
      sg = SignalGroup(digital_signals, label=f'{pid}')
      sg.set_annotation(cls.ANNO_KEY, 30, stages, cls.ANNO_LABELS)
      signal_groups.append(sg)

      # Save sg if necessary
      cls.save_sg_file_if_necessary(pid, sg_path, n_patients, i, sg, **kwargs)

    console.show_status(f'Successfully read {n_patients} files.')
    return signal_groups

  # endregion: Data Loading



if __name__ == '__main__':
  import time

  console.suppress_logging()
  data_dir = r'../../../data/rrsh'

  tic = time.time()
  ds = RRSHSCv1.load_as_sleep_set(data_dir, overwrite=0)

  elapsed = time.time() - tic
  console.show_info(f'Time elapsed = {elapsed:.2f} sec.')

  ds.show()

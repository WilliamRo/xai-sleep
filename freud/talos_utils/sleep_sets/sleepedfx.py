from freud.talos_utils.slp_set import SleepSet
from pictor.objects.signals.signal_group import Annotation
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

  CHANNELS = {'1': 'EEG Fpz-Cz',
              '2': 'EEG Pz-Oz',
              '3': 'EOG horizontal',
              '4': 'Resp oro-nasal',
              '5': 'EMG submental',
              '6': 'Temp rectal',
              '7': 'Event marker'}

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

      # Parse pre-process configs
      pp_configs = kwargs.get('preprocess', '').split(';')
      trim = None
      norm = None
      for config in pp_configs:
        mass = config.split(',')
        if 'trim' in mass[0]:
          assert len(mass) in (1, 2)
          trim = str(30 * 60) if len(mass) == 1 else mass[1]
        elif 'iqr' == mass[0]:
          norm = ('iqr', '1' if len(mass) < 2 else mass[1],
                  '20' if len(mass) < 3 else mass[2])

      # Generate suffix
      suffix = ''
      if trim is not None: suffix += f'trim{trim}'
      if norm is not None:
        if suffix: suffix += ';'
        suffix += f"{','.join(norm)}"

      def _load_raw_sg():
        # If the corresponding .sg file exists, read it directly
        raw_sg_path = os.path.join(data_dir, pid + '(raw)' + '.sg')

        bucket = []
        if cls.try_to_load_sg_directly(
            pid, raw_sg_path, n_patients, i, bucket, **kwargs):
          return bucket[0]

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

        # Save sg if necessary
        cls.save_sg_file_if_necessary(
          pid, raw_sg_path, n_patients, i, sg, **kwargs)
        return sg

      if suffix == '':
        signal_groups.append(_load_raw_sg())
        continue

      # If the corresponding .sg file exists, read it directly
      sg_path = os.path.join(data_dir, pid + f'({suffix})' + '.sg')
      if cls.try_to_load_sg_directly(pid, sg_path, n_patients, i,
                                     signal_groups, **kwargs): continue

      sg = _load_raw_sg()

      # (i) trim wake if required
      if trim is not None:
        trim = float(trim)
        anno: Annotation = sg.annotations[cls.ANNO_KEY]
        ds0 = sg.digital_signals[0]
        # For SleepEDFx data, last interval is usually invalid
        if anno.intervals[-1][0] >= ds0.ticks[-1]:
          anno.intervals.pop(-1)
          anno.annotations = anno.annotations[:-1]

        T1, T2 = 0, ds0.ticks[-1]
        # trim start
        if anno.annotations[0] == 0:
          t1, t2 = anno.intervals[0]
          if t2 - t1 > trim:
            T1 = t2 - trim
            anno.intervals[0] = (T1, t2)

        # trim end
        if anno.annotations[-1] == 0:
          t1, t2 = anno.intervals[-1]
          if t2 - t1 > trim:
            T2 = t1 + trim
            anno.intervals[-1] = (t1, T2)

        for i in range(len(sg.digital_signals)):
          sg.digital_signals[i] = sg.digital_signals[i][T1:T2]

      # (ii) normalize 1st DigitalSignal if required
      if norm is not None:
        if norm[0] == 'iqr':
          iqr, mad = int(norm[1]), int(norm[2])
          for ds in sg.digital_signals: ds.data = DigitalSignal.preprocess_iqr(
            ds.data, iqr=iqr, max_abs_deviation=mad)
        else: raise KeyError(f'!! unknown normalization method {norm[0]}')

      signal_groups.append(sg)
      # Save sg if necessary
      cls.save_sg_file_if_necessary(
        pid, sg_path, n_patients, i, sg, **kwargs)


    console.show_status(f'Successfully read {n_patients} files.')
    return signal_groups

  # endregion: Data Loading



if __name__ == '__main__':
  import time

  console.suppress_logging()
  data_dir = r'../../../data/sleepedfx'

  tic = time.time()
  preprocess = 'trim;iqr'
  ds = SleepEDFx.load_as_sleep_set(data_dir, overwrite=0, preprocess=preprocess)

  elapsed = time.time() - tic
  console.show_info(f'Time elapsed = {elapsed:.2f} sec.')

  ds.show()



import numpy as np
import os
import pandas as pd
import pickle

from pictor.objects.signals.signal_group import DigitalSignal, SignalGroup

from roma.spqr.finder import walk
from roma import console
from roma import io

from xsleep.slp_set import SleepSet
from typing import List

from tframe import DataSet
from tframe.data.sequences.seq_set import SequenceSet


class RRSHSet(SleepSet):
  """The sleep-edf database contains 197 whole-night PolySomnoGraphic sleep
  recordings, containing EEG, EOG, chin EMG, and event markers. Some records
  also contain respiration and body temperature. Corresponding hypnograms
  (sleep patterns) were manually scored by well-trained technicians according
  to the Rechtschaffen and Kales manual, and are also available. """

  TICKS_PER_EPOCH = 100 * 30
  STAGE_KEY = 'STAGE'
  STAGE_LABELS = ['Wake', 'N1', 'N2', 'N3', 'REM', 'Unknown']
  CHANNEL = {'0': 'E1-M2',
             '1': 'E2-M2',
             '2': 'Chin 1-Chin 2',
             '3': 'F3-M2',
             '4': 'C3-M2',
             '5': 'O1-M2',
             '6': 'F4-M1',
             '7': 'C4-M1',
             '8': 'O2-M1',
             '9': 'RIP ECG',
             '10': 'Pleth',
             '11': 'Nasal Pressure',
             '12': 'Therm',
             '13': 'Thor',
             '14': 'Abdo',
             '15': 'Sum',
             '16': 'SpO2',
             '17': 'Snore',
             '18': 'Leg/L',
             '19': 'Leg/R',
             '20': 'PositionSen',
             '21': 'Pulse'
  }

  class DetailKeys:
    number = 'Study Number'
    height = 'Height (cm)'
    weight = 'Weight (kg)'
    gender = 'Gender'
    bmi = 'BMI'
    age = 'Age'
    sleepiness_score = 'Epworth Sleepiness Score'
    study_duration = 'Study Duration (hr)'
    sleep_efficiency = 'Sleep Efficiency (%)'
    num_blocks = 'No of data blocks in EDF'

  # region: Properties

  # endregion: Properties

  # region: Abstract Methods (Data IO)

  @classmethod
  def load_as_tframe_data(cls, data_dir, data_name=None, first_k=None, suffix='',
                          **kwargs) -> SleepSet:
    """...

    suffix list
    -----------
    '': complete dataset
    '-alpha': complete dataset with most wake-signal removed
    """
    suffix_k = '' if first_k is None else f'({first_k})'

    data_dir = os.path.join(data_dir, data_name)
    tfd_path = os.path.join(data_dir, f'{data_name}{suffix_k}{suffix}.tfds')

    # Load .tfd file directly if it exists
    if os.path.exists(tfd_path): return cls.load(tfd_path)

    # Otherwise, wrap raw data into tframe data and save
    console.show_status(f'Loading raw data from `{data_dir}` ...')

    if suffix == '':
      signal_groups = cls.load_raw_data(
        data_dir, save_xai_rec=True, first_k=first_k, **kwargs)
      data_set = RRSHSet(name=f'RRSHSet{suffix_k}',
                         signal_groups=signal_groups)
    elif suffix == '-alpha':
      data_set: RRSHSet = cls.load_as_tframe_data(os.path.dirname(data_dir),
                                                    data_name, first_k)
      data_set.remove_wake_signal(config='terry')
    else: raise KeyError(f'!! Unknown suffix `{suffix}`')

    data_set.save(tfd_path)
    console.show_status(f'Dataset saved to `{tfd_path}`')
    # Save and return
    # io.save_file(data_set, tfd_path, verbose=True)
    return data_set


  @classmethod
  def load_raw_data(cls, data_dir, save_xai_rec=False, first_k=None, **kwargs):
    """Load raw data into signal groups. For each subject, four categories of
    data are read:
    (1) PSG
    (2) Stage labels
    """
    # Sanity check
    import time

    assert os.path.exists(data_dir)

    # Create an empty list
    sleep_groups: List[SignalGroup] = []

    # Get all .XML files
    xml_file_list: List[str] = walk(data_dir, 'file', '*XML*')
    if first_k is not None and first_k != '':
      xml_file_list = xml_file_list[:int(first_k)]
    N = len(xml_file_list)
    print('*' * 20)
    print("patient_num:", N)
    print('*' * 20)
    # Read records in order
    for i, xml_file in enumerate(xml_file_list):
      # Get id
      id: str = os.path.split(xml_file)[-1].split('.')[0]

      # If the corresponding .rec file exists, read it directly
      xai_rec_path = os.path.join(data_dir, id + '.xrec')
      if os.path.exists(xai_rec_path) and not kwargs.get('overwrite', False):
        console.show_status(
          f'Loading `{id}` from {data_dir} ...', prompt=f'[{i + 1}' f'/{N}]')
        console.print_progress(i, N)
        sg = io.load_file(xai_rec_path)
        sleep_groups.append(sg)
        continue

      console.show_status(f'Reading record `{id}` ...', prompt=f'[{i+1}/{N}]')
      console.print_progress(i, N)

      # (1) Read stage labels [0,1,2...]
      stages_anno = cls.read_rrsh_anno_xml(xml_file)
      non_unknown_indice = np.argwhere(np.array(stages_anno) < 5)
      start, end = min(non_unknown_indice)[0], max(non_unknown_indice)[0] + 1
      stages = np.array(stages_anno[start:end])

      # (2) Read PSG file
      fn = os.path.join(data_dir, id + '.edf')
      assert os.path.exists(fn)
      digital_signals: List[DigitalSignal] = cls.read_rrsh_data_mne(
        fn, start=start*cls.TICKS_PER_EPOCH, end=end*cls.TICKS_PER_EPOCH)

      # verify the length of stages and data
      L = (digital_signals[0].length) // cls.TICKS_PER_EPOCH
      assert len(stages) == L

      # Wrap data into signal group
      sg = SignalGroup(digital_signals, label=f'{id}')
      sg.set_annotation(cls.STAGE_KEY, 30, stages, cls.STAGE_LABELS)
      sleep_groups.append(sg)

      # Save sg if necessary
      if save_xai_rec:
        console.show_status(f'Saving `{id}` to `{data_dir}` ...')
        console.print_progress(i, N)
        io.save_file(sg, xai_rec_path)

    console.show_status(f'Successfully read {N} records')
    return sleep_groups

  def configure(self, **kwargs):
    """
    output: [p1,p2,p3,...]
    """
    from tframe import hub as th

    console.show_status(f'configure data...')
    channel_select = kwargs.get('channel_select', None)
    data_name = th.data_config.replace(':', '-') + '-index'
    tfd_path = os.path.join(th.data_dir, 'rrsh', data_name)
    def data_preprocess(data):
      import numpy as np
      from scipy import signal
      #滤波
      b, a= signal.butter(7, 0.7, 'lowpass')
      filted_data = signal.filtfilt(b, a, data)
      #归一化
      arr_mean = np.mean(filted_data)
      arr_std = np.std(filted_data)
      precessed_data = (filted_data - arr_mean) / arr_std
      return precessed_data

    def data_aasm(sg_data, sg_annotation):
      annotation = []
      data_aasm = []
      data_index = []
      sample_length = th.random_sample_length
      for index, label in enumerate(sg_annotation):
        if label in [5]:
          continue
        data_aasm.extend(sg_data[index * sample_length:(index+1) * sample_length])
        annotation.append(label)
        data_index.append(index)
      data_aasm = np.array(data_aasm)
      annotation = np.array(annotation)
      data_index = np.array(data_index)
      return data_aasm, annotation, data_index

    features = []
    targets = []
    data_index_all= []

    if ',' in channel_select:
      chn_names = [self.CHANNEL[i] for i in channel_select.split(',')]
    else:
      chn_names = [self.CHANNEL[channel_select]]

    for sg in self.signal_groups:
      sg_data = np.stack([data_preprocess(sg[name]) for name in chn_names], axis=-1)
      if th.use_rnn:
        sg_data = data_preprocess(sg[chn_names[0]])
      sg_annotation = sg.annotations[self.STAGE_KEY].annotations
      sg_data, sg_annotation, data_index = data_aasm(sg_data, sg_annotation)

      features.append(sg_data)
      targets.append(sg_annotation)
      data_index_all.append(data_index)


    # features[i].shape = [L_i, n, n_channels]
    self.features = features
    # targets[i].shape = [L_i,]
    self.targets = targets
    # data_index[i].shape = [L_i,]
    self.data_index = data_index_all
    console.show_status(f'Finishing configure data...')


  def report(self):
    console.show_info('RRSH Dataset')
    console.supplement(f'Totally {len(self.signal_groups)} subjects', level=2)

  # endregion: Abstract Methods (Data IO)

  # region: Preprocess

  def remove_wake_signal(self, config='terry'):
    assert config == 'terry'

    # For each patient
    for sg in self.signal_groups:
      # Cut annotations
      annotation = sg.annotations[self.STAGE_KEY]
      non_zero_indice = np.argwhere(annotation.annotations != 0)

      start, end = min(non_zero_indice)[0], max(non_zero_indice)[0]

      margin = 60
      start, end = start - margin, end + margin

      annotation.intervals = annotation.intervals[start:end]
      annotation.annotations = annotation.annotations[start:end]

      for ds in sg.digital_signals:
        # TODO
        freq = int(float(ds.label.split('=')[1]))
        _start, _end = start * freq * 30, end * freq * 30
        ds.ticks = ds.ticks[_start:_end]
        ds.sequence = ds.sequence[_start:_end]

  # endregion: Preprocess

  # region: Overwriting

  def _check_data(self):
    """This method will be called during splitting dataset"""
    # assert len(self.signal_groups) > 0
    pass

  # endregion: Overwriting

  # region: Data Visualization

  def show(self):
    from pictor import Pictor
    from pictor.plotters import Monitor
    from tframe import hub as th

    # get validate_data_index [array, array, ...]
    validate_data_index = self.data_index
    validate_pre = 0

    # Initialize pictor and set objects
    p = Pictor(title='RRSHSet', figure_size=(12, 8))
    p.objects = self.signal_groups

    # Set monitor
    channels = ['E2-M2', 'F4-M1', 'C4-M1', 'O2-M1']
    m: Monitor = p.add_plotter(Monitor(channels=','.join(channels)))
    m.channel_list = [c for c, _, _ in self.signal_groups[0].name_tick_data_list]


    # .. set annotation logic
    anno_key = 'annotation'
    anno_str = self.STAGE_KEY + ',prediction'
    m.set(anno_key, anno_str)

    predictions = th.predictions
    if len(predictions) > 0:
      for index, sg in enumerate(self.signal_groups):
        validate_end = validate_data_index[index].shape[0] + validate_pre
        stages = sg.annotations[self.STAGE_KEY].annotations.copy()
        stages[validate_data_index[index]] = predictions[validate_pre:validate_end]
        sg.set_annotation('prediction', 30, stages, RRSHSet.STAGE_LABELS)
        validate_pre = validate_end

    def on_press_a():
      if m.get(anno_key) is None: m.set(anno_key, anno_str)
      else: m.set(anno_key)
    m.register_a_shortcut('a', on_press_a, 'Toggle annotation')

    p.show()

  # endregion: Data Visualization



if __name__ == '__main__':
  from xslp_core import th

  # th.data_config = 'sleepedf'

  th.data_config = 'rrsh:10:1,2,3'
  # _ = UCDDB.load_raw_data(th.data_dir, save_xai_rec=True, overwrite=False)
  data_name, data_num, _ = th.data_config.split(':')
  # SLEEPEDF.load_raw_data(os.path.join(th.data_dir, 'sleepedf'), overwrite=True)
  data_set = RRSHSet.load_as_tframe_data(th.data_dir,
                                         data_name,
                                         data_num)
  data_set.report()
  data_set.show()



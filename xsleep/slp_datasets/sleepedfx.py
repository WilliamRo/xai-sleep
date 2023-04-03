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


class SleepEDFx(SleepSet):
  """The sleep-edf database contains 197 whole-night PolySomnoGraphic sleep
  recordings, containing EEG, EOG, chin EMG, and event markers. Some records
  also contain respiration and body temperature. Corresponding hypnograms
  (sleep patterns) were manually scored by well-trained technicians according
  to the Rechtschaffen and Kales manual, and are also available. """

  CHANNEL = {'0': 'EEG Fpz-Cz',
             '1': 'EEG Pz-Oz',
             '2': 'EOG horizontal',
             '3': 'Resp oro-nasal',
             '4': 'EMG submental',
             '5': 'Temp rectal',
             '6': 'Event marker'}
  ANNO_LABELS = ['Sleep stage W', 'Sleep stage 1', 'Sleep stage 2',
                 'Sleep stage 3', 'Sleep stage 4', 'Sleep stage R',
                 'Movement time', 'Sleep stage ?']
  ANNO_LABELS_AASM = ['Sleep stage W', 'Sleep stage 1', 'Sleep stage 2',
                      'Sleep stage 3', 'Sleep stage R']
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
  def load_as_sleep_set(cls, data_dir, data_name=None, first_k=None,
                        suffix='', **kwargs):
    """...

    suffix list
    -----------
    '': complete dataset
    '-alpha': complete dataset with most wake-signal removed
    """
    suffix_k = 'all' if first_k == '' else f'({first_k})'

    data_dir = os.path.join(data_dir, data_name)
    tfd_path = os.path.join(data_dir, f'{data_name}{suffix_k}{suffix}.tfds')

    # Load .tfd file directly if it exists
    if os.path.exists(tfd_path): return cls.load(tfd_path)

    # Otherwise, wrap raw data into tframe data and save
    console.show_status(f'Loading raw data from `{data_dir}` ...')

    if suffix == '':
      signal_groups = cls.load_as_signal_groups_peiyan(
        data_dir, first_k=first_k, **kwargs)
      data_set = SleepEDFx(name=f'Sleep-EDF-Expanded{suffix_k}',
                           signal_groups=signal_groups)
    elif suffix == '-alpha':
      data_set: SleepEDFx = cls.load_as_sleep_set(
        os.path.dirname(data_dir), data_name, first_k)
      data_set.remove_wake_signal(config='terry')
    else:
      raise KeyError(f'!! Unknown suffix `{suffix}`')

    # data_set.save(tfd_path)
    console.show_status(f'Dataset saved to `{tfd_path}`')
    return data_set

  @classmethod
  def load_as_signal_groups(cls, data_dir, first_k=None, **kwargs):
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
    # Sanity check
    assert os.path.exists(data_dir)

    # Create an empty list
    signal_groups: List[SignalGroup] = []

    # Get all .edf files
    hypnogram_file_list: List[str] = walk(data_dir, 'file', '*Hypnogram*')
    if first_k is not None and first_k != '':
      hypnogram_file_list = hypnogram_file_list[:int(first_k)]
    n_patients = len(hypnogram_file_list)
    # Read records in order
    for i, hypnogram_file in enumerate(hypnogram_file_list):
      # Get id
      id: str = os.path.split(hypnogram_file)[-1].split('-')[0][:7]

      # If the corresponding .sg file exists, read it directly
      sg_path = os.path.join(data_dir, id + '(raw)' + '.sg')
      if cls.try_to_load_sg_directly(id, sg_path, n_patients, i,
                                     signal_groups, **kwargs): continue

      # (1) read annotations
      hypnogram_file = os.path.join(data_dir, hypnogram_file)
      annotation = cls.read_annotations_mne(hypnogram_file, labels=cls.ANNO_LABELS)

      # (2) read PSG file
      fn = os.path.join(data_dir, id + '0' + '-PSG.edf')
      assert os.path.exists(fn)
      digital_signals: List[DigitalSignal] = cls.read_digital_signals_mne(fn)

      # wrap data into signal group
      sg = SignalGroup(digital_signals, label=f'{id}')
      sg.annotations[cls.ANNO_KEY] = annotation
      signal_groups.append(sg)

      # save sg if necessary
      cls.save_sg_file_if_necessary(id, sg_path, n_patients, i, sg, **kwargs)

    console.show_status(f'Successfully read {n_patients} records')
    return signal_groups

  @classmethod
  def load_as_signal_groups_peiyan(cls, data_dir, first_k=None, **kwargs):
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
    # Sanity check
    assert os.path.exists(data_dir)

    # Create an empty list
    signal_groups: List[SignalGroup] = []

    # Get all .edf files
    hypnogram_file_list: List[str] = walk(data_dir, 'file', '*Hypnogram*')
    # hypnogram_file_list: List[str] = [
    #   'SC4151EC-Hypnogram.edf', 'SC4112EC-Hypnogram.edf', 'SC4211EC-Hypnogram.edf',
    #   'SC4281GC-Hypnogram.edf', 'SC4422EA-Hypnogram.edf', 'SC4071EC-Hypnogram.edf',
    #   'SC4081EC-Hypnogram.edf', 'SC4011EH-Hypnogram.edf', 'SC4061EC-Hypnogram.edf',
    #   'SC4651EP-Hypnogram.edf', 'SC4031EC-Hypnogram.edf', 'SC4142EU-Hypnogram.edf',
    #   'SC4102EC-Hypnogram.edf', 'SC4181EC-Hypnogram.edf', 'SC4312EM-Hypnogram.edf',
    #   'SC4001EC-Hypnogram.edf', 'SC4482FJ-Hypnogram.edf', 'SC4021EH-Hypnogram.edf',
    #   'SC4122EV-Hypnogram.edf', 'SC4051EC-Hypnogram.edf',
    # ]
    if first_k is not None and first_k != '':
      hypnogram_file_list = hypnogram_file_list[:int(first_k)]
    n_patients = len(hypnogram_file_list)

    data_file_path = os.path.join(data_dir, 'Sleep_100hz_proposed_denoise_Semi_simulated.npy')
    data_sets = np.load(data_file_path).reshape(20, -1)
    # Read records in order
    for i, hypnogram_file in enumerate(hypnogram_file_list):
      # Get id
      id: str = os.path.split(hypnogram_file)[-1].split('-')[0][:7]

      # If the corresponding .sg file exists, read it directly
      sg_path = os.path.join(data_dir, id + '(Proposed_Semi_simulated)' + '.sg')
      if cls.try_to_load_sg_directly(id, sg_path, n_patients, i,
                                     signal_groups, **kwargs): continue

      # (1) read annotations
      hypnogram_file = os.path.join(data_dir, hypnogram_file)
      annotation = cls.read_annotations_mne(hypnogram_file, labels=cls.ANNO_LABELS)

      # (2) read PSG file
      fn = os.path.join(data_dir, id + '0' + '-PSG.edf')
      assert os.path.exists(fn)
      digital_signals: List[DigitalSignal] = cls.read_digital_signals_mne(fn)
      for ds in digital_signals:
        freq = int(ds.sfreq)
        # ds.data = ds.data[: freq * 78180]
        ds.data = ds.data[: int(freq * 78150)]
        if freq == 100:
          ds.data[:, 0] = data_sets[i][:7815000]

      # wrap data into signal group
      sg = SignalGroup(digital_signals, label=f'{id}')
      sg.annotations[cls.ANNO_KEY] = annotation
      signal_groups.append(sg)

      # save sg if necessary
      # cls.save_sg_file_if_necessary(id, sg_path, n_patients, i, sg, **kwargs)

    console.show_status(f'Successfully read {n_patients} records')
    return signal_groups

  def configure(self, **kwargs):
    """
    features: [ft1, ft2, ft3]
    targets: [tg1, tg2, tg3]
    """
    import numpy as np
    from tframe import hub as th

    channel_select = kwargs.get('channel_select', None)
    console.show_status(f'configure data...')

    features = []
    targets = []
    remove_data_total = []
    select_data_index_total = []
    select_label_index_total = []

    if ',' in channel_select:
      chn_names = [self.CHANNEL[i] for i in channel_select.split(',')]
    else:
      chn_names = [self.CHANNEL[channel_select]]

    for sg in self.signal_groups:
      # configure data
      chn_data = self.interpolate(sg, chn_names)
      for index, data in enumerate(chn_data):
        if index == 0: continue
        else: chn_data[index] = SleepSet.normalize(data)
      sg_data = np.stack([data for data in chn_data], axis=-1)

      # configure annotation
      annotations = np.array(sg.annotations[self.ANNO_KEY].annotations)
      intervals = sg.annotations[self.ANNO_KEY].intervals
      sg_annotation = []
      for index, interval in enumerate(intervals):
        sg_annotation.extend(np.ones(int(interval[1] - interval[0]) // 30, dtype=np.int)
                             * annotations[index])

      # convert to aasm format
      sg_data, sg_annotation, remove_data, data_index, label_index = self.clean_data(sg_data, sg_annotation)
      sg_annotation = np.where(sg_annotation == 4, 3, sg_annotation)
      sg_annotation = np.where(sg_annotation == 5, 4, sg_annotation)
      sg.annotations[self.ANNO_KEY].labels = self.ANNO_LABELS_AASM

      features.append(sg_data)
      targets.append(sg_annotation)
      select_data_index_total.append(data_index)
      select_label_index_total.append(label_index)
      remove_data_total.extend(remove_data)

    if th.add_noise:
      self.add_noise(features, targets, chn_names, remove_data_total, select_label_index_total)

    # features[i].shape = [L_i, n_channels]
    self.features = features
    # targets[i].shape = [L_i,]
    self.targets = targets
    self.unknown = remove_data_total
    self.label_index = select_label_index_total

    console.show_status(f'Finishing configure data...')

  def clean_data(self, data, annotation):
    """
    remove and save unknown data
    """
    from tframe import hub as th

    epoch_length = th.random_sample_length
    assert data.shape[0] == len(annotation) * epoch_length

    # clean outlier

    remove_label_index = np.argwhere(np.array(annotation) > 5)
    remove_data_index = []
    for index in remove_label_index:
      data_index = int(index * epoch_length) + np.arange(epoch_length, dtype=np.int)
      remove_data_index.extend(data_index)
    select_data_index = np.setdiff1d(np.arange(len(data)), remove_data_index)
    select_label_index = np.setdiff1d(np.arange(len(annotation)), remove_label_index)
    assert len(select_label_index) == len(select_data_index) // epoch_length
    select_data = data[select_data_index]
    select_label = np.array(annotation)[select_label_index]
    remove_data = data[remove_data_index]
    if len(remove_data) >= epoch_length:
      remove_data = np.split(remove_data, len(remove_data) // epoch_length)
    return select_data, select_label, remove_data, select_data_index, select_label_index

  def report(self):
    console.show_info('Sleep-EDFx Dataset')
    console.supplement(f'Totally {len(self.signal_groups)} subjects',
                       level=2)

  # endregion: Abstract Methods (Data IO)

  def _check_data(self):
    """This method will be called during splitting dataset"""
    # assert len(self.signal_groups) > 0
    pass

  # region: Data Visualization

  def show_old(self):
    from pictor import Pictor
    from pictor.plotters import Monitor
    from tframe import hub as th


    # Initialize pictor and set objects
    p = Pictor(title='Sleep-EDFx', figure_size=(12, 8))
    p.objects = self.signal_groups

    # Set monitor
    channels = ['EEG Fpz-Cz', 'EEG Pz-Oz', 'EOG horizontal',
                'Resp oro-nasal']
    m: Monitor = p.add_plotter(Monitor(channels=','.join(channels)))
    m.channel_list = [c for c, _, _ in
                      self.signal_groups[0].name_tick_data_list]

    # .. set annotation logic
    anno_key = 'annotation'
    anno_str = self.STAGE_KEY + ',prediction'
    m.set(anno_key, anno_str)

    predictions = np.array(th.predictions)
    if len(predictions) > 0:
      validate_data_index = self.label_index
      validate_pre = 0
      for index, sg in enumerate(self.signal_groups):
        validate_end = validate_data_index[index].shape[0] + validate_pre
        stages = sg.annotations[self.STAGE_KEY].annotations.copy()
        stages[validate_data_index[index]] = predictions[validate_pre:validate_end]
        sg.set_annotation('prediction', 30, stages,
                          SleepEDFx.STAGE_LABELS)

        validate_pre = validate_end

    def on_press_a():
      if m.get(anno_key) is None:
        m.set(anno_key, self.STAGE_KEY)
      else:
        m.set(anno_key)

    m.register_a_shortcut('a', on_press_a, 'Toggle annotation')

    p.show()

  def show(self, *funcs, **kwargs):
    from freud.gui.freud_gui import Freud

    # Initialize pictor and set objects
    freud = Freud(title=str(self.__class__.__name__))
    freud.objects = self.signal_groups

    for func in [func for func in funcs if callable(func)]: func(freud.monitor)

    for k, v in kwargs.items(): freud.monitor.set(k, v, auto_refresh=False)
    freud.show()
  # endregion: Data Visualization


if __name__ == '__main__':
  # th.data_config = 'sleepedfx'
  from xslp_core import th

  data_config = 'sleepedfx:10:0,1,2'
  # _ = UCDDB.load_raw_data(th.data_dir, save_xai_rec=True, overwrite=False)
  data_name, data_num, channel_select = data_config.split(':')
  # SLEEPEDF.load_raw_data(os.path.join(th.data_dir, 'sleepedfx'), overwrite=True)
  data_set = SleepEDFx.load_as_sleep_set(th.data_dir,
                                         data_name,
                                         data_num,
                                         suffix='')
  data_set.show()

from freud.talos_utils.slp_config import SleepConfig
from pictor.objects.signals.digital_signal import DigitalSignal
from pictor.objects.signals.signal_group import SignalGroup, Annotation
from roma import io
from tframe.data.sequences.seq_set import SequenceSet, DataSet
from tframe.utils.misc import convert_to_one_hot
from tframe import console
from typing import List

import numpy as np
import os



class SleepSet(SequenceSet):

  class Keys:
    tapes = 'SleepSet::Keys::tapes'
    map_dict = 'SleepSet::Keys::map_dict'
    epoch_tables = 'SleepSet::Keys::epoch_table'

  ANNO_KEY_GT_STAGE = 'stage Ground-Truth'

  EPOCH_DURATION = 30.0

  CHANNELS = {}
  NUM_STAGES = 5

  AASM_LABELS = ['Wake', 'N1', 'N2', 'N3', 'REM', '?']

  # valid data-kwargs in th.data_config
  VALID_KWARGS = ['val_ids',
                  'test_ids',
                  'preprocess',
                  'mad',                  # max absolute deviation (IQR)
                  'sg_preprocess']

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

    return self.extract_data_set(include_targets=True)

  @SequenceSet.property()
  def epoch_table(self):
    """Epoch table will not be generated before it is called first-time.
    This table contains NUM_STAGES + 1 (typically 6) lists, each of which
    contains a list of tuples comprising information of a period of stage,
    i.e., table[<STAGE_ID>] = [(sg, start_t, duration), ...]. For example,
    table[0] contains all wake stage information from all signal groups.
    """
    table = [[] for _ in range(self.NUM_STAGES + 1)]
    for sg in self.signal_groups:
      for i in range(self.NUM_STAGES + 1):
        table[i].extend(self.get_sg_epoch_tables(sg)[0][i])
    return table

  # endregion: Properties

  # region: Overwriting

  def _check_data(self): pass

  # region: gen_batches

  # region: Multi-epoch sampling

  def _sample_seqs_from_sg(self, sg, start_time, duration, with_stage=False):
    """Sample a sequence from a signal group.

    :param sg - signal group.
    :param start_time - start time in seconds.
    :param duration - in seconds.

    :return (data, label) if with_stage. Otherwise, return data.
            data.shape = [L, C]; label.shape = [E],
            here E is epoch number, equals to L / fs / 30.
    """
    from tframe import hub as th
    assert isinstance(th, SleepConfig)

    # For now consider only 1 branch TODO
    assert len(th.fusion_channels) == 1

    # Only one tape (branch) is considered for now
    # tape.shape = [L_tape, C]
    tape, fs = sg.get_from_pocket(self.Keys.tapes)[0]

    start_i, L = int(start_time * fs), int(duration * fs)
    assert L < tape.shape[0]
    assert start_i % 30 == 0   # TODO: this assertion is for sleep staging only

    # Make sure start_i is legal
    start_i = min(max(0, start_i), tape.shape[0] - L)
    data = tape[start_i:start_i+L]

    if with_stage:
      stage_ids = self.get_sg_epoch_tables(sg)[1]
      start_j, L = int(start_i / fs / 30), int(duration / 30)
      labels = stage_ids[start_j:start_j+L]
      return data, labels

    return data

  def _get_sequence_randomly(self, batch_size):
    """Randomly samples a DataSet, whose features.shape = [B, L, C],
       targets.shape = [B, E, 5], here L = E * ticks_per_epoch.
       Note features.shape can be easily reshaped to [B, E, L/E, C].
       Unlike single epoch sampling, class balance is more difficult to
       achieve when sampling a sequence.
    """
    from tframe import hub as th
    assert isinstance(th, SleepConfig)

    features, targets = [], []

    # epoch_table = [[(sg, start_t, duration), ...], ...]
    for sid in np.random.randint(0, self.NUM_STAGES, batch_size):
      table = self.epoch_table[sid]
      sg, start_t, duration = table[np.random.randint(0, len(table))]

      MAX_COUNT = 100
      for count in range(MAX_COUNT):
        data, labels = self._sample_seqs_from_sg(
          sg, start_t, th.epoch_num * 30, with_stage=True)
        if None not in labels: break
      if None in labels: raise AssertionError(
        f'!! Failed to sample valid data after {MAX_COUNT} attempts.')

      features.append(data)
      # TODO: ? class is not considered for now
      t = convert_to_one_hot(labels, self.NUM_STAGES)
      t = np.reshape(t, [th.epoch_num, self.NUM_STAGES])
      targets.append(t)

    features = np.stack(features, axis=0)
    targets = np.stack(targets, axis=0)
    return DataSet(features, targets, NUM_CLASSES=self.NUM_STAGES)

  # endregion: Multi-epoch sampling

  # region: Single-epoch sampling

  def _get_branches_randomly(self, batch_size: int):
    """Currently only 1 branch is supported"""
    from tframe import hub as th
    assert isinstance(th, SleepConfig)

    if th.use_rnn: raise NotImplementedError(
      '!! RNN inputs are not supported currently')

    # Generate FNN inputs
    branches = [[] for _ in th.fusion_channels]
    assert len(branches) == 1
    stage_ids = []

    # epoch_table = [[(sg, start_t, duration), ...], ...]
    # For each record in this batch, first decide its corresponding stage.
    for sid in np.random.randint(0, self.NUM_STAGES, batch_size):
      table = self.epoch_table[sid]
      # Then choose a random interval
      sg, start_t, duration = table[np.random.randint(0, len(table))]

      # Get tape and fs
      for branch, tape_tuple in zip(
          branches, sg.get_from_pocket(self.Keys.tapes)):
        tape, fs = tape_tuple
        # Sliding-window augmentation, default epoch_delta = 0.2
        start_t += (np.random.rand() * 2 - 1) * duration * th.epoch_delta
        d = int(duration * fs)
        start_i = min(max(int(start_t * fs), 0), len(tape) - d)
        branch.append(tape[start_i:start_i+d])
        stage_ids.append(sid)

    # Generate features and targets
    features = np.stack(branches[0], axis=0)
    targets = convert_to_one_hot(stage_ids, self.NUM_STAGES)

    return DataSet(features, targets, NUM_CLASSES=5)

  # endregion: Single-epoch sampling

  def gen_batches(self, batch_size, shuffle=False, is_training=False):
    """Generate FNN batches (xs, ys). Each xs[i] has a shape of [E, L, C].
    Here C represents number of channels, E is the number of epochs,
    and L = fs * 30, where fs is sampling rate and E. Each ys[i] has a shape of
    [E, 5].
    """
    from tframe import hub as th

    # Validate training set
    if not is_training:
      for batch in self.validation_set.gen_batches(batch_size): yield batch
      return

    round_len = self.get_round_length(batch_size, training=is_training)
    assert batch_size != -1

    # Generate batches
    for i in range(round_len):
      if th.epoch_num > 1:
        data_batch = self._get_sequence_randomly(batch_size)
      else:
        data_batch = self._get_branches_randomly(batch_size)

      # Preprocess if necessary
      if self.batch_preprocessor is not None:
        data_batch = self.batch_preprocessor(data_batch, is_training)
      # Make sure data_batch is a regular array
      if not data_batch.is_regular_array: data_batch = data_batch.stack
      # Yield data batch
      yield data_batch

    # Clear dynamic_round_len
    self._clear_dynamic_round_len()

  # endregion: gen_batches

  @property
  def size(self): return len(self.validation_set.features)

  # endregion: Overwriting

  # region: Standards

  @classmethod
  def get_map_dict(cls, sg: SignalGroup):
    # TODO: currently only AASM standard is supported
    assert cls.NUM_STAGES == 5
    anno: Annotation = sg.annotations[cls.ANNO_KEY_GT_STAGE]

    def _init_map_dict(labels):
      map_dict = {}
      # TODO: this will be called for each sg, which is verbosing
      # console.show_status('Creating mapping ...')
      for i, label in enumerate(labels):
        if 'W' in label: j = 0
        elif '1' in label: j = 1
        elif '2' in label: j = 2
        elif '3' in label or '4' in label: j = 3
        elif 'R' in label: j = 4
        else: j = None
        map_dict[i] = j
        # console.supplement(f'{label} maps to {j}', level=2)
      return map_dict

    return sg.get_from_pocket(
      cls.Keys.map_dict, initializer=lambda: _init_map_dict(anno.labels))

  @classmethod
  def get_sg_epoch_tables(cls, sg: SignalGroup):
    """[Rule] 0: Wake, 1: N1, 2: N2, 3: N3, 4: REM, None: Unknown
       (1) table_per_class = [[(sg, start_i, duration), ...], ...]
       (2) table_id = [0, 1, 2, ...]
    """
    def _init_sg_epoch_tables():
      # Get annotation
      anno: Annotation = sg.annotations[cls.ANNO_KEY_GT_STAGE]
      # Generate map_dict
      map_dict = cls.get_map_dict(sg)
      # 5 stages + 1 unknown label
      table_per_class = [[] for _ in range(cls.NUM_STAGES + 1)]
      table_id = []

      t0 = anno.intervals[0][0]
      for interval, anno_id in zip(anno.intervals, anno.annotations):
        sid = map_dict[anno_id]
        N = (interval[-1] - interval[0]) / cls.EPOCH_DURATION
        # Check N
        assert N == int(N)

        # The 1st element in tuple is for future concatenating
        table_per_class[sid if sid is not None else 5].extend([
          (sg, interval[0] + i * cls.EPOCH_DURATION - t0, cls.EPOCH_DURATION)
          for i in range(int(N))])
        table_id.extend([sid] * int(N))
      return table_per_class, table_id

    return sg.get_from_pocket(
      cls.Keys.epoch_tables, initializer=_init_sg_epoch_tables)

  # endregion: Standards

  # region: Methods for configuration

  def configure(self):
    """Configure dataset.
       th.data_config example:
       (1) 'sleepedfx 1,2;3'
    """
    from tframe import hub as th
    assert isinstance(th, SleepConfig)

    # Check data-kwargs
    for key in th.data_kwargs:
      if key not in self.VALID_KWARGS: raise KeyError(
        f'!! `{key}` is not a valid data-kwargs')

    # (0) Set class names for ConfusionMatrix
    self.properties['CLASSES'] = ['Wake', 'N1', 'N2', 'N3', 'REM']

    # (1) extract required channels as tapes according to channel selection
    console.show_status('Extracting tapes ...')
    for i, sg in enumerate(self.signal_groups):
      console.print_progress(i, len(self.signal_groups))

      if sg.in_pocket(self.Keys.tapes): continue

      tapes = []
      # fusion_channels = [['1', '2'], ['3']]
      for chn_lst in th.fusion_channels:
        # tape.shape = [L, C], TODO: fusion channels should have the same sfreq
        tape = np.stack([sg[self.CHANNELS[key]] for key in chn_lst], axis=-1)

        # Get corresponding sfreq
        sfreq = sg.channel_signal_dict[self.CHANNELS[chn_lst[0]]].sfreq
        tapes.append((tape, sfreq))

      sg.put_into_pocket(self.Keys.tapes, tapes)

  # endregion: Methods for configuration

  # region: Public Methods

  def get_subset_by_patient_id(self, indices, name_suffix='subset'):
    return self.__class__(
      name=f'{self.name}-{name_suffix}',
      signal_groups=[self.signal_groups[i] for i in indices],
      CLASSES=self['CLASSES'], NUM_CLASSES=self.NUM_STAGES)

  def extract_data_set(self, include_targets=False):
    """Extract talos.DataSet from self.signal_groups based on th.
       Note that self.configure method should be called beforehand. """
    from tframe import hub as th
    assert isinstance(th, SleepConfig)

    # Generate FNN inputs
    branches = [[] for _ in th.fusion_channels]
    stage_ids = []
    for sg in self.signal_groups:
      tapes = sg.get_from_pocket(self.Keys.tapes)
      for branch, tape_tuple in zip(branches, tapes):
        tape, sfreq = tape_tuple
        assert isinstance(tape, np.ndarray)

        # tape.shape = [L, C], default epoch_num is 1
        ticks_per_seq = int(self.EPOCH_DURATION * sfreq) * th.epoch_num

        # Truncate tape if necessary
        L = tape.shape[0] // ticks_per_seq * ticks_per_seq
        x = tape[:L].reshape([-1, ticks_per_seq, tape.shape[-1]])

        branch.append(x)
      # Find stage_ids if necessary
      if include_targets: stage_ids.extend(self.get_sg_epoch_tables(sg)[1])

    # TODO currently only single branch is supported
    assert len(branches) == 1
    features = np.concatenate(branches[0], axis=0)

    # TODO: here len(ids) may > len(branches[i]), e.g., in SleepEDFx data
    data_dict = {}
    if include_targets:
      stage_ids = stage_ids[:len(features)]
      mask = [(0 if si is None else 1) for si in stage_ids]
      data_dict['mask'] = np.stack(mask, axis=-1)

      # Removed invalid epochs
      #nan_indices = [i for i, v in enumerate(stage_ids) if v is None]
      # features = np.delete(features, nan_indices, axis=0)
      # targets = [v for v in stage_ids if v is not None]

      targets = [0 if si is None else si for si in stage_ids]
      targets = convert_to_one_hot(targets, self.NUM_STAGES)
    else: targets = None

    # NUM_CLASSES and CLASSES properties are for confusion matrix label
    return DataSet(features, targets, data_dict, name=f'{self.name}-val',
                   NUM_CLASSES=self.NUM_STAGES, CLASSES=self['CLASSES'])

  # endregion: Public Methods

  # region: Abstract Methods

  @classmethod
  def load_as_signal_groups(cls, data_dir, **kwargs) -> List[SignalGroup]:
    raise NotImplementedError

  @classmethod
  def load_as_sleep_set(cls, data_dir, **kwargs) -> SequenceSet:
    try:
      from tframe import hub as th
      kwargs['preprocess'] = th.data_kwargs.get('sg_preprocess', '')
    except: pass

    sg = cls.load_as_signal_groups(data_dir, **kwargs)
    return cls(name=cls.__name__, signal_groups=sg, NUM_CLASSES=cls.NUM_STAGES)

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

    if kwargs.get('return_freud', False): return freud

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

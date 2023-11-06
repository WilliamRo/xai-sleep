from freud.gui.sleep_monitor import SleepMonitor
from roma import console
from freud.data_io.mne_based import read_digital_signals_mne
from freud.gui.sleep_monitor import SleepMonitor
from pictor import Pictor
from pictor.objects.signals.signal_group import SignalGroup, Annotation
from pictor.plugins import DialogUtilities
from roma import console

import os
import numpy as np
class Freud(Pictor, DialogUtilities):

  def __init__(self, title='Freud', figure_size=(12, 7)):
    super(Freud, self).__init__(title, figure_size=figure_size)

    self.monitor = self.add_plotter(SleepMonitor(self))

  # region: Commands

  def open(self, edf_path: str = None, dtype=float, auto_refresh=True):
    """Open an EDF file. If `edf_path` is not provided, an `open_file` dialog
    will be popped up for manually selecting file."""
    if edf_path is None:
      edf_path = self.load_file_dialog('Please select an EDF file')
    if edf_path in ('', ): return

    fn = os.path.basename(edf_path)

    with self.busy(f'Reading data from `{edf_path}` ...', auto_refresh):
      digital_signals = read_digital_signals_mne(edf_path, dtype=dtype)
      sg = SignalGroup(digital_signals, label=f'{fn}')
      self.objects.append(sg)

    # Refresh if necessary
    if auto_refresh: self.refresh()

  def standardize_stage_annotation(self, standard='aasm', auto_refresh=True):
    from freud.talos_utils.slp_set import SleepSet

    assert standard == 'aasm'
    ANNO_KEY = 'stage Ground-Truth'
    for sg in self.objects:
      assert isinstance(sg, SignalGroup)
      if ANNO_KEY not in sg.annotations: continue

      map_dict = SleepSet.get_map_dict(sg)
      for k in list(map_dict.keys()):
        if map_dict[k] is None: map_dict[k] = 6

      anno: Annotation = sg.annotations[ANNO_KEY]
      anno.labels = SleepSet.AASM_LABELS
      anno.annotations = [map_dict[a] for a in anno.annotations]

      console.show_status(f'Stages in `{sg.label}` has been standardized.')

    if auto_refresh: self.refresh()
  ssa = standardize_stage_annotation

  def modify_annotation(self, standard='alpha', auto_refresh=True):

    '''
    modify the annotation according to AASM
    such as
    '''
    from freud.talos_utils.slp_set import SleepSet
    assert standard == 'alpha'
    ANNO_KEY = 'stage Ground-Truth'

    for sg in self.objects:

      if ANNO_KEY not in sg.annotations: continue

      for key, value in sg.annotations.items():
        annotations = value.annotations

        # else:
        if 'dsn' in key:
          import copy
          Wake2REM, N32REM = 0, 0

          sg.annotations['stage ma'] = copy.deepcopy(value)
          # value = sg.annotations['stage ma']
          for i in range(1, len(sg.annotations['stage ma'].annotations)):
            if annotations[i-1] == 0 and annotations[i] == 4:
              sg.annotations['stage ma'].annotations[i] = 0
              Wake2REM +=1
            elif annotations[i-1] == 3 and annotations[i] == 4:
              sg.annotations['stage ma'].annotations[i] = 3
              N32REM += 1
          print('wake to REM have appeared and changed {} times'.format(Wake2REM))
          print('N3 to REM have appeared and changed {} times'.format(N32REM))

          self.compare(sg.annotations['stage dsn_lite'],
                       sg.annotations['stage Ground-Truth'],
                       name='dsn_lite')
          self.compare(sg.annotations['stage ma'],
                       sg.annotations['stage Ground-Truth'],
                       name='ma')
  ma = modify_annotation

  def report(self, standard = 'alpha'):
    assert standard == 'alpha'
    ANNO_KEY = 'stage Ground-Truth'
    import  numpy as np
    # result = np.zeros(5)
    all_result = []
    for sg in self.objects:
      result = np.zeros(5)
      if ANNO_KEY not in sg.annotations: continue
      for key, value in sg.annotations.items():
        print('in the {}'.format(key))
        annotations = value.annotations
        if 'Ground-Truth' in key:

          # for anno in value.annotations:
          for i in range(1, len(annotations)):
            if annotations[i - 1] == 0 and annotations[i] == 4: result[0] += 1
            if annotations[i - 1] == 3 and annotations[i] == 4: result[1] += 1
            if annotations[i - 1] == 2 and annotations[i] == 1: result[2] += 1
            if annotations[i - 1] == 1 and annotations[i] == 2: result[3] += 1

          print('wake to REM have appeared {} times'.format(result[0]))
          print('N3 to REM have appeared {} times'.format(result[1]))
          print('N2 to N1 have appeared {} times'.format(result[2]))
          print('N1 to N2 have appeared {} times'.format(result[3]))
          print("-----------------------------------------------")
          all_result.append(result)


  def compare(self, pred_annotations:Annotation,
              gt_annotations:Annotation,**kwargs):
      from tframe.utils import console

      def process_annotations(annotations):
        """
        transfer annotations to the method,
         which each epoch corresponds one by one
        """
        labels = []
        for i, interval in enumerate(annotations.intervals):
          index_start = int(interval[0])
          index_end = int(interval[1])
          epoch_len = int((index_end - index_start) / 30)
          labels.append([annotations.annotations[i]] * epoch_len)
        return np.concatenate(labels)

      pred_labels = process_annotations(pred_annotations)
      gt_labels = process_annotations(gt_annotations)


      # get the confusionMatrix
      from tframe.utils.maths.confusion_matrix import ConfusionMatrix
      cm = ConfusionMatrix(
        num_classes=5,
        class_names=['W', '1', '2', '3', 'R'])
      cm.fill(pred_labels, gt_labels)
      name = kwargs.get('name', None)
      console.show_info('Confusion Matrix:{}'.format(name))
      console.write_line(cm.make_table(
        decimal=4, class_details=True))

      return cm


  # endregion: Commands

  # region: APIs

  @staticmethod
  def visualize_signal_groups(signal_groups, title='Freud',
                              figure_size=(12, 8), **kwargs):
    fre = Freud(title, figure_size)
    for k, v in kwargs.items(): fre.monitor.set(k, v, auto_refresh=False)
    fre.objects = signal_groups
    fre.show()

  # endregion: APIs

if __name__ == '__main__':
  from roma import finder
  from roma import io

  # Set directories
  data_dir = r'../../data/'
  data_dir += 'sleepeasonx'

  prefix = ['', 'sleepedfx', 'ucddb', 'rrsh'][1]
  pattern = f'{prefix}*.sg'
  pattern = f'*SC4[01]*'

  # Select .sg files
  sg_file_list = finder.walk(data_dir, pattern=pattern)

  signal_groups = []
  for path in sg_file_list:
    sg = io.load_file(path, verbose=True)
    signal_groups.append(sg)

  # Visualize signal groups
  Freud.visualize_signal_groups(signal_groups, figure_size=(11, 6),
                                default_win_duration=999999)
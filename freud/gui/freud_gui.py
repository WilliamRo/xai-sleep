from freud.data_io.mne_based import read_digital_signals_mne
from freud.gui.sleep_monitor import SleepMonitor
from pictor import Pictor
from pictor.objects.signals.signal_group import SignalGroup
from pictor.plugins import DialogUtilities

import os
import numpy as np



class Freud(Pictor, DialogUtilities):

  def __init__(self, title='Freud', figure_size=(12, 8)):
    super(Freud, self).__init__(title, figure_size=figure_size)

    self.monitor = self.add_plotter(SleepMonitor())

  # region: Commands

  def open(self, edf_path: str = None, dtype=np.float, auto_refresh=True):
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

  # endregion: Commands



if __name__ == '__main__':
  freud = Freud()
  freud.show()

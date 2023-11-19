from epoch_explorer_base import EpochExplorer, RhythmPlotter
from freud.data_io.mne_based import read_digital_signals_mne
from freud.gui.sleep_monitor import SleepMonitor
from pictor import Pictor
from pictor.plotters.plotter_base import Plotter
from pictor.objects.signals.signal_group import SignalGroup, Annotation
from pictor.plugins import DialogUtilities
from roma import console

import os
import matplotlib.pyplot as plt
import numpy as np



class RhythmPlotterPro(RhythmPlotter):

  def __init__(self, pictor, **kwargs):
    super().__init__(self.plot, pictor, **kwargs)

    # Define settable attributes
    # self.new_settable_attr('xxx', True, bool, 'Whether to plot wave')

  # region: Plot Methods



if __name__ == '__main__':
  from roma import finder
  from roma import io

  # Set directories
  data_dir = r'../../data/'
  data_dir += 'sleepeasonx'

  prefix = ['', 'sleepedfx', 'ucddb', 'rrsh'][1]
  pattern = f'{prefix}*.sg'

  # Select .sg files
  sg_file_list = finder.walk(data_dir, pattern=pattern)[:20]

  signal_groups = []
  for path in sg_file_list:
    sg = io.load_file(path, verbose=True)
    signal_groups.append(sg)

  # Visualize signal groups
  EpochExplorer.explore(signal_groups, plot_wave=True)



import copy

from epoch_explorer_base import EpochExplorer, RhythmPlotter
from freud.data_io.mne_based import read_digital_signals_mne
from freud.gui.sleep_monitor import SleepMonitor
from matplotlib.backend_tools import Cursors
from matplotlib.patches import Rectangle
from matplotlib.widgets import RectangleSelector
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
    super().__init__(pictor, **kwargs)

    # Define settable attributes
    # self.new_settable_attr('xxx', True, bool, 'Whether to plot wave')


  # region: Omics

  # region: Core Methods

  def _calc_freqs(self, signals: np.ndarray):
    freqs = []
    for i, s in enumerate(signals):
      # console.print_progress(i, len(signals))
      f, secs, spectrum, dom_f = self._calc_dominate_freq_curve_v1(s)
      # Estimate freq score based on dom_f
      score = np.average(dom_f)
      freqs.append(score)

    return freqs

  def _calc_amps(self, signals: np.ndarray):
    amps = []
    for i, s in enumerate(signals):
      # console.print_progress(i, len(signals))
      upper, lower = self.pooling(s, int(self.get('dev_arg')))
      # Estimate amp score based on upper and lower
      score = np.average(upper - lower)
      amps.append(score)

    return amps

  # endregion: Core Methods

  def analyze_selected_sg(self, channel_index: int = 0,
                          show_region='true'):
    """
    channel_index: ...
    show_region: option to show rectangular bounds
    """
    # (1) Find se
    sg = self.explorer.selected_signal_group
    se = self.explorer.get_sg_stage_epoch_dict(sg)
    # se.keys() == ['W', 'N1', 'N2', 'N3', 'R']
    # se.values() is a list of signals with shape [fs * 30， C]
    # selected_signal = se[STAGE_KEY][EPOCH_INDEX][:, CHANNEL_INDEX]

    # (2) Calculate fre and amp scores for each epoch for all stages
    colors = {           # see https://matplotlib.org/stable/gallery/color/named_colors.html
      'W': 'forestgreen', 'N1': 'gold', 'N2': 'orange', 'N3': 'royalblue',
      'R': 'lightcoral'
    }
    results = {}
    for _, stage_key in enumerate(colors.keys()):
      if stage_key not in se: continue
      signals = [data[:, channel_index] for data in se[stage_key]]
      console.show_status(f'Estimating scores for `{stage_key}` ...')
      frequencies = self._calc_freqs(signals)
      amplitudes = self._calc_amps(signals)
      results[stage_key] = (frequencies, amplitudes)
      console.show_status(f'{len(signals)} epochs estimated.')

    # (3) set results to sg for future filtering
    fa_scores = results
    sg.put_into_pocket('fa_scores', fa_scores, exclusive=False)

    # (*) Plot
    fig: plt.Figure = plt.figure()
    ax: plt.Axes = fig.add_subplot(111)

    for stage_key, fas in results.items():
      freqs, amps = results[stage_key]
      color = colors[stage_key]
      ax.scatter(freqs, amps, c=color, label=stage_key, alpha=0.5)

      # show region if required
      if show_region.lower() in ('true', '1'):
        f_min, f_max = np.min(freqs), np.max(freqs)
        a_min, a_max = np.min(amps), np.max(amps)
        rect = Rectangle((f_min, a_min), f_max - f_min, a_max - a_min,
                         alpha=0.5, fill=False, edgecolor=color)
        ax.add_patch(rect)

    # .. set styles and show
    channel_name = sg.digital_signals[0].channels_names[channel_index]

    minF, maxF = self.get('min_freq'), self.get('max_freq')
    amp_size = int(self.get('dev_arg'))
    title = f'[{sg.label}][{channel_name}]'
    title += f' fre: ({minF}, {maxF}); amp-size: {amp_size}'
    ax.set_title(title)
    ax.set_xlabel('Frequency Score')
    ax.set_ylabel('Amplitude Score')
    ax.legend()

    # .. add region selector function
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    def onselect_function(*_):
      xmin, xmax, ymin, ymax = rect_selector.extents
      # self.put_into_pocket('fa_rect', rect_selector.extents, exclusive=False)
      rect = Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                       facecolor='red', edgecolor='black',
                       alpha=0.2, fill=True)

      # Filter and explore
      new_sg = copy.deepcopy(sg)
      new_se = {}
      for s_key, data in se.items():
        f_scores, a_scores = results[s_key]
        data = [s for s, f, a in zip(data, f_scores, a_scores)
                if xmin < f < xmax and ymin < a < ymax]
        if len(data) > 0: new_se[s_key] = data
      if len(new_se) == 0: return

      # Add patch and refresh canvas
      ax.add_patch(rect)
      fig.canvas.draw()

      # Create a new epoch explorer and show selected epochs
      new_sg.put_into_pocket(EpochExplorer.Keys.STAGE_EPOCH_DICT, new_se,
                             exclusive=False)
      EpochExplorer.explore([new_sg], plot_wave=True,
                            plotter_cls=RhythmPlotterPro)
    rect_selector = RectangleSelector(ax, onselect_function, button=[1])
    ax.set_xlim(xlim), ax.set_ylim(ylim)

    # .. adjust and show
    fig.subplots_adjust()
    fig.show()

  ss = analyze_selected_sg

  # endregion: Omics



if __name__ == '__main__':
  from roma import finder
  from roma import io

  # Set directories
  data_dir = r'../../data/'
  data_dir += 'sleepeasonx'
  # data_dir += 'sleepedfx'

  prefix = ['', 'sleepedfx', 'ucddb', 'rrsh'][1]
  pattern = f'{prefix}*.sg'
  # pattern = f'SC*raw*.sg'

  channel_names = ['EEG Fpz-Cz', 'EEG Pz-Oz']

  # Select .sg files
  sg_file_list = finder.walk(data_dir, pattern=pattern)[:20]

  signal_groups = []
  for path in sg_file_list:
    sg: SignalGroup = io.load_file(path, verbose=True)
    if channel_names: sg = sg.extract_channels(channel_names)
    signal_groups.append(sg)

  # Visualize signal groups
  EpochExplorer.explore(signal_groups, plot_wave=True,
                        plotter_cls=RhythmPlotterPro)



from matplotlib.patches import Rectangle
from pictor import Pictor
from pictor.plotters.plotter_base import Plotter
from roma import console

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas



class FPViewer(Pictor):

  class Keys(Pictor.Keys):
    CHANNELS = 'ChAnNeLs'

  def __init__(self, title='Epoch Walker Viewer', figure_size=(10, 6), **kwargs):
    # Call parent's constructor
    super(FPViewer, self).__init__(title, figure_size=figure_size)

    self.probe_scatter = self.add_plotter(ProbeScatter(self))
    self.data = {}

    if 'walker_results' in kwargs:
      self.load_walker_results(kwargs['walker_results'])

    self.shortcuts._library.pop('Escape')

  # region: Properties

  @property
  def bm_arg_tuples(self): return [
    (bm_key, arg_key) for bm_key, (arg_key, _) in self.data['meta'][2].items()]

  @Pictor.property()
  def dataframe_dict(self):
    """key = (sg.label, channel_label)"""
    import pandas as pd

    stage_keys = ('W', 'N1', 'N2', 'N3', 'R')

    series_dict, stage_key_dict = {}, {}
    for data_key, stage_dict in self.data.items():
      if data_key == 'meta': continue
      sg_label, ch_label, config_tuple = data_key
      # Initialize stage_key_series if necessary
      if sg_label not in stage_key_dict:
        sk_list = []
        for sk in stage_keys: sk_list.extend([sk] * len(stage_dict[sk]))
        stage_key_dict[sg_label] = pd.Series(data=sk_list, name='Stage')

      group_key = (sg_label, ch_label)

      # Initialize if necessary, group_key = ('sg.label', 'channel_label')
      if group_key not in series_dict:
        series_dict[group_key] = {'Stage': stage_key_dict[sg_label]}

      value_list = []
      for sk in stage_keys: value_list.extend(stage_dict[sk])
      bm_key, param_key, param_value = config_tuple
      series_key = f'{bm_key} ({param_key}={param_value})'
      series_dict[group_key][series_key] = pd.Series(
        data=value_list, name=series_key)

    # Gather data into DataFrames
    df_dict = {}
    for s_key, s_dict in series_dict.items():
      df_dict[s_key] = pd.DataFrame(data=s_dict)

    return df_dict

  @property
  def selected_profiles(self):
    # e.g., 'sleepedfSC4001E'
    sg_label = self.get_element(self.Keys.OBJECTS)
    # e.g., 'EEG Fpz-Cz'
    channel_label = self.get_element(self.Keys.CHANNELS)
    key = [sg_label, channel_label]
    profiles = []
    for bm_key, arg_key in self.bm_arg_tuples:
      # e.g., bm_key = 'BM01-FREQ', arg_key = 'max_freq'
      # e.g., arg = 25
      arg = self.get_element((bm_key, arg_key))
      # e.g., profile = ('SC4001E', 'EEG Fpz-Cz', (BM01, 'max_freq', 25))
      profiles.append(tuple(key + [(bm_key, arg_key, arg)]))

    return profiles

  @property
  def selected_res_dict(self):
    return {key[2][0]: self.data[key] for key in self.selected_profiles}

  @property
  def selected_dataframe(self):
    # e.g., 'sleepedfSC4001E'
    sg_label = self.get_element(self.Keys.OBJECTS)
    # e.g., 'EEG Fpz-Cz'
    channel_label = self.get_element(self.Keys.CHANNELS)

    return self.dataframe_dict[(sg_label, channel_label)]

  # endregion: Properties

  # region: Public Methods

  def load_walker_results(self, results: dict):
    """- results['meta'] = (
           sg_label_list,
           channel_list,
           bm_args_dict     # {'BM01': ('arg1', [32, 64]), ...}
         )
       - results[(<sg_label>, <channel>, (<bm_key>, <arg>, <value>)] =
         {<stage_key>: bm_output_list}
    """
    sg_label_list, channel_list, bm_arg_dict = results['meta']

    self.set_to_axis(self.Keys.OBJECTS, sg_label_list, overwrite=True)
    self.set_to_axis(self.Keys.CHANNELS, channel_list)

    axis_keys = []
    for bm_key, (arg_key, arg_list) in bm_arg_dict.items():
      axis_key = (bm_key, arg_key)
      # self.create_dimension(axis_key)
      self.set_to_axis(axis_key, arg_list)
      axis_keys.append(axis_key)

    self.data = results

    self._register_key('l', 'Next arg for BM-1', axis_keys[0], 1)
    self._register_key('h', 'Previous arg for BM-1', axis_keys[0], -1)
    self._register_key('j', 'Next arg for BM-2', axis_keys[1], 1)
    self._register_key('k', 'Previous arg for BM-2', axis_keys[1], -1)

  # endregion: Public Methods

  # region: Shortcuts

  def _register_key(self, btn, des, key, v):
    self.shortcuts.register_key_event(
      [btn], lambda: self.set_cursor(key, v, refresh=True),
      description=des, color='yellow')

  def _register_default_key_events(self):
    # Allow plotter shortcuts
    self.shortcuts.external_fetcher = self._get_plotter_shortcuts

    register_key = self._register_key

    register_key('N', 'Next Signal Group', self.Keys.OBJECTS, 1)
    register_key('P', 'Previous Signal Group', self.Keys.OBJECTS, -1)

    register_key('greater', 'Next Plotter', self.Keys.PLOTTERS, 1)
    register_key('less', 'Previous Plotter', self.Keys.PLOTTERS, -1)

    register_key('n', 'Next Channel', self.Keys.CHANNELS, 1)
    register_key('p', 'Previous Channel', self.Keys.CHANNELS, -1)

  # endregion: Shortcuts


class ProbeScatter(Plotter):

  def __init__(self, pictor, **kwargs):
    super().__init__(self.plot, pictor)

    self.new_settable_attr('show_scatter', True, bool,
                           'Option to show scatter')
    self.new_settable_attr('show_rect', False, bool,
                           'Option to show region of each stage')
    self.new_settable_attr('show_kde', True, bool,
                           'Option to show KDE for each stage')
    self.new_settable_attr('show_vector', False, bool,
                           'Option to show KDE for each stage')

    self.new_settable_attr('xmin', None, float, 'x-min')
    self.new_settable_attr('xmax', None, float, 'x-max')
    self.new_settable_attr('ymin', None, float, 'y-min')
    self.new_settable_attr('ymax', None, float, 'y-max')
    self.new_settable_attr('scatter_alpha', 0.5, float, 'scatter_alpha')

    self.new_settable_attr('margin', 0.15, float, 'margin')

  def register_shortcuts(self):
    self.register_a_shortcut('s', lambda: self.flip('show_scatter'),
                             'Toggle `show_scatter`')
    self.register_a_shortcut('r', lambda: self.flip('show_rect'),
                             'Toggle `show_rect`')
    self.register_a_shortcut('g', lambda: self.flip('show_kde'),
                             'Toggle `show_kde`')
    self.register_a_shortcut('v', lambda: self.flip('show_vector'),
                             'Toggle `show_vector`')

  # region: Plot Methods

  def plot(self, ax: plt.Axes, fig: plt.Figure):
    """
    res_dict = {<bm_key>: {'W': array_w, 'N1': array_n1, ...}, ...}
    """
    res_dict: dict = self.pictor.selected_res_dict

    assert len(res_dict) == 2

    colors = {           # see https://matplotlib.org/stable/gallery/color/named_colors.html
      'W': 'forestgreen', 'N1': 'gold', 'N2': 'orange', 'N3': 'royalblue',
      'R': 'lightcoral'
    }

    bm1_key, bm2_key = list(res_dict.keys())
    xmin, xmax, ymin, ymax = np.inf, -np.inf, np.inf, -np.inf
    for stage_key, color in colors.items():
      if stage_key not in res_dict[bm1_key]: continue
      data1, data2 = res_dict[bm1_key][stage_key], res_dict[bm2_key][stage_key]

      if len(data1) < 2: continue

      # Convert data2 to micro-volt
      data2 = data2 * 1e6

      if self.get('show_scatter'):
        alpha = self.get('scatter_alpha')
        ax.scatter(data1, data2, c=color, label=stage_key, alpha=alpha)

      # show region if required
      # if self.get('show_rect'): self.show_bounds(ax, data1, data2, color)

      # show gauss is required
      if self.get('show_kde'): self.show_kde(ax, data1, data2, color)

      # show vector is required
      if self.get('show_vector'): self.show_vector(ax, data1, data2, color)

      # Update limits
      xmin, xmax = min(xmin, np.min(data1)), max(xmax, np.max(data1))
      ymin, ymax = min(ymin, np.min(data2)), max(ymax, np.max(data2))

    # Set title, axis labels, and legend
    profiles = self.pictor.selected_profiles
    sg_label, channel_key = profiles[0][:2]
    ax.set_title(f'{sg_label} ({channel_key})')

    bm_key, arg_key, arg_v = profiles[0][2]
    ax.set_xlabel(f'{bm_key} ({arg_key}={arg_v})')

    m = self.get('margin')
    xm, ym = (xmax - xmin) * m, (ymax - ymin) * m
    xmin, xmax = xmin - xm, xmax + xm
    ymin, ymax = ymin - ym, ymax + ym
    xmin = self.get('xmin') if self.get('xmin') is not None else xmin
    xmax = self.get('xmax') if self.get('xmax') is not None else xmax
    ymin = self.get('ymin') if self.get('ymin') is not None else ymin
    ymax = self.get('ymax') if self.get('ymax') is not None else ymax
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # if self.get('xmin') is not None:
    #   ax.set_xlim(self.get('xmin'), self.get('xmax'))
    #   ax.set_ylim(self.get('ymin'), self.get('ymax'))
    # else:
    #   m = self.get('margin')
    #   xm, ym = (xmax - xmin) * m, (ymax - ymin) * m
    #   ax.set_xlim(xmin - xm, xmax + xm)
    #   ax.set_ylim(ymin - ym, ymax + ym)

    bm_key, arg_key, arg_v = profiles[1][2]
    ax.set_ylabel(f'{bm_key} ({arg_key}={arg_v})')
    ax.legend()


  def movie(self):
    amp_key, freq_key = ('BM02-AMP', 'pool_size'), ('BM01-FREQ', 'max_freq')
    fre_inc = lambda: self.pictor.set_cursor(freq_key, 1, refresh=True)
    amp_inc = lambda: self.pictor.set_cursor(amp_key, 1, refresh=True)
    scripts = []
    for _ in range(5):
      scripts.append(amp_inc)
      scripts.append(fre_inc)
    scripts.append(amp_inc)
    # scripts.append(amp_inc)
    self.pictor.animate(fps=2, scripts=scripts)

  # endregion: Plot Methods

  # region: Data Analysis


  def show_vector(self, ax: plt.Axes, m1, m2, color):
    mu1, mu2 = np.mean(m1), np.mean(m2)
    # Calculate covariance matrix
    cov = np.cov(m1, m2)
    assert cov[0, 1] == cov[1, 0]
    k = cov[0, 1] / cov[0, 0]
    x1, y1 = mu1, mu2
    step = np.sqrt(cov[0, 0])
    x2, y2 = mu1 + step, mu2 + step * k

    ax.plot(x1, y1, 's', color=color)
    ax.plot([x1, x2], [y1, y2], '-', color=color)

  def show_kde(self, ax: plt.Axes, m1, m2, color):
    from scipy import stats

    xmin, xmax = np.min(m1), np.max(m1)
    ymin, ymax = np.min(m2), np.max(m2)

    # Set margin
    m = self.get('margin')
    xm, ym = (xmax - xmin) * m, (ymax - ymin) * m
    xmin, xmax = xmin - xm, xmax + xm
    ymin, ymax = ymin - ym, ymax + ym

    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([m1, m2])
    kernel = stats.gaussian_kde(values)
    Z = np.reshape(kernel(positions).T, X.shape)

    ax.contour(X, Y, Z, colors=color)

  def show_bounds(self, ax: plt.Axes, data1, data2, color):
    f_min, f_max = np.min(data1), np.max(data1)
    a_min, a_max = np.min(data2), np.max(data2)
    rect = Rectangle((f_min, a_min), f_max - f_min, a_max - a_min,
                     alpha=0.5, fill=False, edgecolor=color)
    ax.add_patch(rect)

  # endregion: Data Analysis



if __name__ == '__main__':
  from tframe.utils.file_tools.io_utils import load

  save_path = r'P:\xai-sleep\data\probe_reports\sg10_eeg2_bm01(15,35)_bm02(32,224).pr'
  save_path = r'P:\xai-sleep\data\probe_reports\sg10_eeg2_bm01(15,40)_bm02(32,256).pr'
  save_path = r'P:\xai-sleep\data\probe_reports\sg10_eeg2_bm01(25)_bm02(128).pr'
  save_path = r'P:\xai-sleep\data\probe_reports\rrsh_insomnia_eeg4_bm01(25)_bm02(128).pr'
  save_path = r'P:\xai-sleep\data\probe_reports\rrsh_narcolepsy_eeg4_bm01(25)_bm02(128).pr'
  results = load(save_path)
  meta = results['meta']

  ew = EWViewer(walker_results=results)
  ew.show()
  print()





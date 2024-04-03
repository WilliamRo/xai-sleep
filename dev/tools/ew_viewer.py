from matplotlib.patches import Rectangle
from pictor import Pictor
from pictor.plotters.plotter_base import Plotter
from roma import console

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas



class EWViewer(Pictor):

  class Keys(Pictor.Keys):
    CHANNELS = 'ChAnNeLs'

  def __init__(self, title='Epoch Walker Viewer', figure_size=(10, 6), **kwargs):
    # Call parent's constructor
    super(EWViewer, self).__init__(title, figure_size=figure_size)

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
    self.new_settable_attr('fit_gauss', False, bool,
                           'Option to fit gauss for each stage')

    self.new_settable_attr('xmin', None, float, 'x-min')
    self.new_settable_attr('xmax', None, float, 'x-max')
    self.new_settable_attr('ymin', None, float, 'y-min')
    self.new_settable_attr('ymax', None, float, 'y-max')

  def register_shortcuts(self):
    self.register_a_shortcut('f', self.fpa, '...')

    self.register_a_shortcut('s', lambda: self.flip('show_scatter'),
                             'Toggle `show_scatter`')
    self.register_a_shortcut('r', lambda: self.flip('show_rect'),
                             'Toggle `show_rect`')
    self.register_a_shortcut('g', lambda: self.flip('fit_gauss'),
                             'Toggle `fit_gauss`')

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
    for stage_key, color in colors.items():
      if stage_key not in res_dict[bm1_key]: continue
      data1, data2 = res_dict[bm1_key][stage_key], res_dict[bm2_key][stage_key]

      if self.get('show_scatter'):
        ax.scatter(data1, data2, c=color, label=stage_key, alpha=0.5)

      # show region if required
      if self.get('show_rect'): self.show_bounds(ax, data1, data2, color)

      # show gauss is required
      # if self.get('fit_gauss'): self.fit_gauss(ax, data1, data2, color)

    # Set title, axis labels, and legend
    profiles = self.pictor.selected_profiles
    sg_label, channel_key = profiles[0][:2]
    ax.set_title(f'{sg_label} ({channel_key})')

    bm_key, arg_key, arg_v = profiles[0][2]
    ax.set_xlabel(f'{bm_key} ({arg_key}={arg_v})')

    ax.set_xlim(self.get('xmin'), self.get('xmax'))
    ax.set_ylim(self.get('ymin'), self.get('ymax'))

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

  def finger_print_alpha(self):
    """
    res_dict = {<bm_key>: {'W': array_w, 'N1': array_n1, ...}, ...}
    """
    import seaborn as sns
    import pandas as pd

    # sns.set_theme()
    sns.set_palette(['forestgreen', 'gold', 'orange', 'royalblue',
                     'lightcoral'])

    # Get dataframe
    df: pd.DataFrame = self.pictor.selected_dataframe

    (_, _, cfg1), (_, _, cfg2) = self.pictor.selected_profiles
    keys = [f'{cfg[0]} ({cfg[1]}={cfg[2]})' for cfg in (cfg1, cfg2)]

    sns.displot(df, x=keys[0], y=keys[1], hue='Stage', kind='kde')

    plt.show()

  fpa = finger_print_alpha

  def export_fpa(self, tgt_path=None, overwrite=False, yscale: float = 1.0,
                 xlim=None, ylim=None):
    import seaborn as sns
    import pandas as pd
    import warnings

    # Ignore all warnings
    warnings.filterwarnings("ignore")

    # Check path to export
    if tgt_path is None:
      from pictor.plugins.dialog_utils import DialogUtilities
      tgt_path = DialogUtilities.select_folder_dialog(
        'Please select path to export')
    if not tgt_path: return
    assert os.path.exists(tgt_path)
    console.show_status(f'Target path set to `{tgt_path}`.')

    # Get all sg_labels and channels
    v: EWViewer = self.pictor
    sg_label_list, channel_list, _ = v.data['meta']

    # Set keys as selected profiles
    (_, _, cfg1), (_, _, cfg2) = self.pictor.selected_profiles
    keys = [f'{cfg[0]} ({cfg[1]}={cfg[2]})' for cfg in (cfg1, cfg2)]

    # Export fingerprints of all sg across all channels
    N, i = len(sg_label_list) * len(channel_list), 0
    for sg_label in sg_label_list:
      console.show_status(f'Analyzing PID {sg_label} ...')
      for channel in channel_list:
        console.show_status(f'Exporting fingerprints of channel {channel} ...')
        console.print_progress(i, N)

        file_name = f'{sg_label},{channel}.png'
        file_path = os.path.join(tgt_path, file_name)

        if os.path.exists(file_path) and not overwrite:
          N -= 1
          continue

        df: pd.DataFrame = v.dataframe_dict[(sg_label, channel)]

        # Set color
        palette = []
        stages = df['Stage'].tolist()
        if 'W' in stages: palette.append('forestgreen')
        if 'N1' in stages: palette.append('gold')
        if 'N2' in stages: palette.append('orange')
        if 'N3' in stages: palette.append('royalblue')
        if 'R' in stages: palette.append('lightcoral')
        sns.set_palette(palette)

        # Generate KDE plot
        p = sns.displot(df, x=keys[0], y=keys[1], hue='Stage', kind='kde')
        if yscale != 1.0:
          formatter = matplotlib.ticker.FuncFormatter(
            lambda x, pos: f'{x * yscale:.0f}')
          p.axes.flat[0].yaxis.set_major_formatter(formatter)

        # Set styles
        plt.title(f'PID: {sg_label}, Channel: {channel}')

        # Export figure
        plt.tight_layout()
        plt.savefig(file_path)

        i += 1

    console.show_status(f'Successfully exported {N} fingerprints.')

  fpa = export_fpa


  def show_displot(self, channel_index=0):
    from roma import console

    import seaborn as sns
    import pandas as pd

    # sns.set_theme()
    sns.set_palette(['forestgreen', 'gold', 'orange', 'royalblue',
                     'lightcoral'])

    # Traverse all sg

    # Get dataframe
    df: pd.DataFrame = self.pictor.selected_dataframe

    (_, _, cfg1), (_, _, cfg2) = self.pictor.selected_profiles
    keys = [f'{cfg[0]} ({cfg[1]}={cfg[2]})' for cfg in (cfg1, cfg2)]

    g = sns.displot(df, x=keys[0], y=keys[1], hue='Stage', kind='kde')

    # g = sns.displot(df, x=keys[0], hue='Stage', kind='kde')

    # p = sns.displot(df, x=keys[1], hue='Stage', kind='kde')
    # formatter = matplotlib.ticker.FuncFormatter(
    #   lambda x, pos: f'{x * 1e6:.0f}')
    # p.axes.flat[0].xaxis.set_major_formatter(formatter)

    plt.show()
  disp = show_displot


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





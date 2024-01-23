import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from pictor import Pictor
from pictor.plotters.plotter_base import Plotter
from roma import console

import numpy as np



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

  @property
  def selected_profiles(self):
    sg_label = self.get_element(self.Keys.OBJECTS)
    channel_label = self.get_element(self.Keys.CHANNELS)
    key = [sg_label, channel_label]
    profiles = []
    for bm_key, arg_key in self.bm_arg_tuples:
      arg = self.get_element((bm_key, arg_key))
      profiles.append(tuple(key + [(bm_key, arg_key, arg)]))

    return profiles

  @property
  def selected_res_dict(self):
    return {key[2][0]: self.data[key] for key in self.selected_profiles}

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

    self.new_settable_attr('show_rect', True, bool,
                           'Option to show region of each stage')

    self.new_settable_attr('xmin', None, float, 'x-min')
    self.new_settable_attr('xmax', None, float, 'x-max')
    self.new_settable_attr('ymin', None, float, 'y-min')
    self.new_settable_attr('ymax', None, float, 'y-max')

  def plot(self, ax: plt.Axes):
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
      ax.scatter(data1, data2, c=color, label=stage_key, alpha=0.5)

      # show region if required
      if self.get('show_rect'):
        f_min, f_max = np.min(data1), np.max(data1)
        a_min, a_max = np.min(data2), np.max(data2)
        rect = Rectangle((f_min, a_min), f_max - f_min, a_max - a_min,
                         alpha=0.5, fill=False, edgecolor=color)
        ax.add_patch(rect)

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



if __name__ == '__main__':
  from tframe.utils.file_tools.io_utils import load

  save_path = r'P:\xai-sleep\data\probe_reports\sg10_eeg2_bm01(15,35)_bm02(32,224).pr'
  save_path = r'P:\xai-sleep\data\probe_reports\sg10_eeg2_bm01(15,40)_bm02(32,256).pr'
  results = load(save_path)
  meta = results['meta']

  ew = EWViewer(walker_results=results)
  ew.show()
  print()





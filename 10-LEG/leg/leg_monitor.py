from freud.gui.sleep_monitor import SleepMonitor
from roma.spqr.arguments import Arguments
import matplotlib.pyplot as plt
import numpy as np

from pictor.objects.signals.signal_group import SignalGroup, Annotation
from pictor.objects.signals.scrolling import Scrolling



class LegMonitor(SleepMonitor):

  def __init__(self, pictor=None, window_duration=60, channels: str='*'):
    """
    :param window_duration: uses second as unit
    """
    # Call parent's constructor
    super(LegMonitor, self).__init__(pictor=pictor, window_duration=window_duration, channels=channels)

    # Specific attributes
    self._leg_annotations_to_show = {}

    self.new_settable_attr('leg', False, bool, '...')

  def register_shortcuts(self):
    super(LegMonitor, self).register_shortcuts()

    self.register_a_shortcut('A', lambda: self.flip('leg'),
                             'Whether to show leg movement annotation')


  def _plot_annotation(self, ax: plt.Axes, s: Scrolling):
    axes_dict, kwargs = {}, {}
    legend_handles = []
    for i, anno_str in enumerate(self._annotations_to_show):
      anno_config = Arguments.parse(anno_str)
      key: str = anno_config.func_name
      if key.lower() in ('sleep_stage', 'stage'):
        plot_method = self._plot_stage
        kwargs['index'] = i
      else: raise KeyError(f'!! Unknown annotation key `{key}`')

      # Try to fetch package
      start_time, end_time = ax.get_xlim()
      package = s.get_annotation(anno_str, start_time, end_time)
      if package is None: continue

      # Get results
      right_ax, line = plot_method(
        ax, axes_dict.get(key, None), package, anno_config, **kwargs)
      axes_dict[key] = right_ax
      # Set label to line
      label = anno_config.arg_list[0]
      line.set_label(label)
      legend_handles.append(line)

    # TODO: -----------------------------------------------------
    if self.get('leg'):
      # if not(self._leg_annotations_to_show):
      #   self.mark_leg_move(['Leg/L','Leg/R'])
      self.plot_events(['Leg/L', 'Leg/R'], 'leg_move', ax, s)

    # TODO: -----------------------------------------------------

    # Show legend if necessary
    if len(legend_handles) > 1 or self.get('anno_legend'):
      ax.legend(handles=legend_handles, framealpha=1.0).set_zorder(99)


  def plot_events(self, channel_keys: str, anno_key: str, ax: plt.Axes, s):

    for a, channel_key in enumerate(channel_keys):
      # get y,x
      channels = self.get('channels')
      channels = channels.split(',')
      N = len(channels)
      y = N - channels.index(channel_key) - 1

      # data = s.get_channels(channel_key, max_ticks=self.get('max_ticks'))
      # x = data[0][1]

      # get marker
      annos = self._leg_annotations_to_show

      colorlist = ['red', 'blue', 'green', 'yellow']
      for i, anno_name in enumerate(annos):
        starts = annos[anno_name][channel_key]['start']
        durations = annos[anno_name][channel_key]['duration']

        for num, anno in enumerate(zip(starts,durations)):
          rect = plt.Rectangle((anno[0],y),anno[1],1,facecolor=colorlist[i],fill=True,alpha=0.25)
          ax.add_patch(rect)
        # ax.legend()

  pe = plot_events


  def mark_leg_move(self,channel_keys):
    from leg.leg_move_marker import marker_alpha
    from leg.leg_move_marker import marker_beta
    # 1. prepare data
    sg: SignalGroup = self._selected_signal
    self._leg_annotations_to_show['ground_truth']={}
    self._leg_annotations_to_show['alpha']={}

    for a, channel_key in enumerate(channel_keys):
      self._leg_annotations_to_show['ground_truth'][channel_key] = marker_beta(sg,channel_key)
      self._leg_annotations_to_show['alpha'][channel_key] = marker_alpha(sg,channel_key)

  mlm = mark_leg_move

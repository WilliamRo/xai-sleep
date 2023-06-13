from freud.gui.sleep_monitor import SleepMonitor
from roma.spqr.arguments import Arguments
import matplotlib.pyplot as plt
import numpy as np

from pictor.objects.signals.signal_group import SignalGroup, Annotation
from pictor.objects.signals.scrolling import Scrolling



class LegMonitor(SleepMonitor):

  def _plot_annotation(self, ax: plt.Axes, s: Scrolling):
    axes_dict, kwargs = {}, {}
    legend_handles = []
    for i, anno_str in enumerate(self._annotations_to_show):
      anno_config = Arguments.parse(anno_str)
      key: str = anno_config.func_name

      # Try to fetch package
      start_time, end_time = ax.get_xlim()
      package = s.get_annotation(anno_str, start_time, end_time)
      if package is None: continue

      # Get results
      if key.lower() in ('sleep_stage', 'stage'):
        kwargs['index'] = i
        right_ax, line = self._plot_stage(
          ax, axes_dict.get(key, None), package, anno_config, **kwargs)
        axes_dict[key] = right_ax
        # Set label to line
        label = anno_config.arg_list[0]
        line.set_label(label)
        legend_handles.append(line)

      elif key.lower() in ('event'):
        kwargs['index'] = i
        if package.intervals: self._plot_event(ax, package, anno_config, **kwargs)

      else:
        raise KeyError(f'!! Unknown annotation key `{key}`')


    # Show legend if necessary
    if len(legend_handles) > 1 or self.get('anno_legend'):
      ax.legend(handles=legend_handles, framealpha=1.0).set_zorder(99)


  def _plot_event(self, ax: plt.Axes, package, config: Arguments, index):

    channel_key = package.labels

    # get y,x
    channels = self.get('channels')
    channels = channels.split(',')
    N = len(channels)
    y = N - channels.index(channel_key) - 1

    # get marker
    annos = package.intervals

    colorlist = ['red', 'blue', 'green', 'yellow']
    for anno in annos:
      rect = plt.Rectangle((anno[0],y),(anno[1] - anno[0]),1,facecolor='blue',fill=True,alpha=0.25)
      ax.add_patch(rect)
      # ax.legend()

  pe = _plot_event


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

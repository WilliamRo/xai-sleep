from freud.gui.sleep_monitor import SleepMonitor
from pictor.objects.signals.scrolling import Scrolling
from roma.spqr.arguments import Arguments

import matplotlib.pyplot as plt




class LegMonitor(SleepMonitor):

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
    self.plot_event('Leg/L', 'leg_move', ax, s)
    # TODO: -----------------------------------------------------

    # Show legend if necessary
    if len(legend_handles) > 1 or self.get('anno_legend'):
      ax.legend(handles=legend_handles, framealpha=1.0).set_zorder(99)


  def plot_event(self, channel_key: str, anno_key: str, ax: plt.Axes, s):
    pass
  pe = plot_event


  def mark_leg_move(self):
    from leg.leg_move_marker import marker_alpha
  mlm = mark_leg_move

from freud.gui.sleep_monitor import SleepMonitor
from roma.spqr.arguments import Arguments
from roma import console
from pictor.objects.signals.signal_group import SignalGroup, Annotation
from pictor.objects.signals.scrolling import Scrolling

import matplotlib.pyplot as plt
import time



class LegMonitor(SleepMonitor):
  """This plotter is used in combination with RRSHv1 dataset"""

  # region: Annotation Plotter

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
      elif key.lower() in ('event', 'event-auto'):
        color = 'blue' if 'auto' not in key.lower() else 'red'
        self._plot_event(ax, package, color=color)
      else:
        raise KeyError(f'!! Unknown annotation key `{key}`')

    # Show legend if necessary
    if len(legend_handles) > 1 or self.get('anno_legend'):
      ax.legend(handles=legend_handles, framealpha=1.0).set_zorder(99)

  def _plot_event(self, ax: plt.Axes, package, color='blue'):
    # Find which channel to draw events
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
      rect = plt.Rectangle(
        (anno[0], y), (anno[1] - anno[0]), 1,
        facecolor=color, fill=True, alpha=0.25)
      ax.add_patch(rect)
      # ax.legend()

  pe = _plot_event

  # endregion: Annotation Plotter

  # region: Auto Marking Methods

  def mark_leg_move(self, leg='l', verbose=True):
    from leg.leg_move_marker import mark_single_channel_alpha

    assert leg in ('l', 'r')
    key = 'Leg/L' if leg == 'l' else 'Leg/R'

    x, y = self._selected_signal.name_tick_data_dict[key]
    tic = time.time()
    interval_indices = mark_single_channel_alpha(y)
    if verbose:
      elapsed = time.time() - tic
      console.show_info(f'Time elapsed for mark_leg_move = {elapsed:.2f} sec.')

    intervals = [(x[i1], x[i2]) for i1, i2 in interval_indices]
    anno = Annotation(intervals, labels=key)
    anno_key = f'event-auto {key}-alpha'
    self._selected_signal.annotations[anno_key] = anno
    self.toggle_annotation(*anno_key.split(' '))

    # from leg.leg_move_marker import marker_alpha
    # from leg.leg_move_marker import marker_beta
    # # 1. prepare data
    # sg: SignalGroup = self._selected_signal
    # self._leg_annotations_to_show['ground_truth']={}
    # self._leg_annotations_to_show['alpha']={}
    #
    # for a, channel_key in enumerate(channel_keys):
    #   self._leg_annotations_to_show['ground_truth'][channel_key] = marker_beta(sg,channel_key)
    #   self._leg_annotations_to_show['alpha'][channel_key] = marker_alpha(sg,channel_key)

  mlm = mark_leg_move

  # endregion: Auto Marking Methods

  # region: Useful Commands

  def next_prev_leg_event(self, direction):
    """direction should be -1 or 1"""
    assert direction in (-1, 1)
    # Get current start time
    ss = self._selected_signal
    _, ticks, _ = ss.get_channels('Leg/L')[0]
    T0 = ticks[0]

    # Find next leg event
    keys = ['event Limb-Movement-(Left)', 'event Limb-Movement-(Right)']
    intervals = []
    for k in keys: intervals += ss.annotations[k].intervals
    intervals = sorted(intervals, key=lambda x: x[0], reverse=direction==-1)

    gap = 1
    for t0, _ in intervals:
      if direction == 1:
        if t0 > T0 + gap: break
      else:
        if t0 < T0 - gap: break

    self.goto(t0)

  def register_shortcuts(self):
    super(LegMonitor, self).register_shortcuts()

    self.register_a_shortcut('N', lambda : self.next_prev_leg_event(1),
                             description='Find next leg movement event')
    self.register_a_shortcut('P', lambda : self.next_prev_leg_event(-1),
                             description='Find previous leg movement event')

  # endregion: Useful Commands

from freud.gui.sleep_monitor import SleepMonitor
from roma.spqr.arguments import Arguments
from roma import console
from pictor.objects.signals.signal_group import SignalGroup, Annotation
from pictor.objects.signals.scrolling import Scrolling

import matplotlib.pyplot as plt
import time



class LegMonitor(SleepMonitor):
  """This plotter is used in combination with RRSHv1 dataset"""

  GT_L_LEG_KEY = 'event Limb-Movement-(Left)'
  GT_R_LEG_KEY = 'event Limb-Movement-(Right)'

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
      elif key.lower() in ('event', 'event_auto'):
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

  def leg_move_evaluation(self, leg='L', report=False, alpha=0.5):
    from leg.leg_move_evaluation import leg_move_evaluation
    dt_key = f'event_auto Leg/{leg}-alpha'
    keys = [self.GT_L_LEG_KEY, dt_key]
    gt_intervals, dt_intervals = [
      self._selected_signal.annotations[k].intervals for k in keys]

    # Try to find results in pocket
    res_key = dt_key + '-metrics'
    results = self.get_from_pocket(
      res_key, initializer=lambda: leg_move_evaluation(
        gt_intervals, dt_intervals, report=report))

    # Print metrics if required
    if report:
      Ndt, Ngt = len(dt_intervals), len(gt_intervals)
      console.show_info(f'Evaluation Metrics (alpha = {alpha})')
      console.supplement(
        f'{Ndt} events detected, TP = {results[0]}, GT# = {Ngt}', level=2)
      console.supplement(f'Precision = {results[1]:.3f}', level=2)
      console.supplement(f'Recall = {results[2]:.3f} ', level=2)

    return results

  def mark_leg_move(self, leg='l', verbose=True):
    from leg.leg_move_marker import mark_single_channel_alpha
    assert leg in ('l', 'r')
    key = 'Leg/L' if leg == 'l' else 'Leg/R'

    x, y = self._selected_signal.name_tick_data_dict[key]
    tic = time.time()
    interval_indices = mark_single_channel_alpha(y, self._selected_signal.dominate_signal.sfreq)
    if verbose:
      elapsed = time.time() - tic
      console.show_info(f'Time elapsed for mark_leg_move = {elapsed:.2f} sec.')

    intervals = [(x[i1], x[i2]) for i1, i2 in interval_indices]
    anno = Annotation(intervals, labels=key)
    anno_key = f'event_auto {key}-alpha'
    self._selected_signal.annotations[anno_key] = anno
    self.toggle_annotation(*anno_key.split(' '), force_on=True)

    if self.GT_L_LEG_KEY in self._annotations_to_show:
      self.leg_move_evaluation(leg='L', report=True)

  mlm = mark_leg_move

  # endregion: Auto Marking Methods

  # region: Useful Commands

  def _next_prev_interval(self, direction, intervals):
    """direction should be -1 or 1"""
    assert direction in (-1, 1)

    # Get current start time
    ss = self._selected_signal
    _, ticks, _ = ss.get_channels('Leg/L')[0]
    T0 = ticks[0]

    gap = 1
    for t0, _ in intervals:
      if direction == 1:
        if t0 > T0 + gap: break
      else:
        if t0 < T0 - gap: break

    self.goto(t0)

  def next_prev_leg_event(self, direction):
    # Get intervals
    keys = [self.GT_L_LEG_KEY, self.GT_R_LEG_KEY]
    intervals = []
    for k in keys:
      intervals += self._selected_signal.annotations[k].intervals
    intervals = sorted(intervals, key=lambda x: x[0], reverse=direction==-1)

    self._next_prev_interval(direction, intervals)

  def next_prev_marker_error(self, direction, list_name='local_error_list'):
    from leg.leg_move_evaluation import leg_move_evaluation

    # Find next leg event
    results = self.leg_move_evaluation(leg='L')

    if list_name == 'local_error_list': intervals = results[5]
    elif list_name == 'FP_list': intervals = results[4]

    self._next_prev_interval(direction, intervals)

  def register_shortcuts(self):
    super(LegMonitor, self).register_shortcuts()

    self.register_a_shortcut('N', lambda : self.next_prev_leg_event(1),
                             description='Find next leg movement event')
    self.register_a_shortcut('P', lambda : self.next_prev_leg_event(-1),
                             description='Find previous leg movement event')

    self.shortcuts['L'] = (lambda : self.mark_leg_move('l'),
                           'marker leg/left', 'yello')

    def toggle_gt_events():
      keys = [self.GT_L_LEG_KEY, self.GT_R_LEG_KEY]
      for i, k in enumerate(keys):
        self.toggle_annotation(*k.split(' '), auto_refresh=i==len(keys)-1)

    self.register_a_shortcut('a', toggle_gt_events,
                             description='Toggle ground-truth events')

    self.register_a_shortcut('v', lambda: self.next_prev_marker_error(1),
                             description='Find next local error')
    self.register_a_shortcut('c', lambda: self.next_prev_marker_error(-1),
                             description='Find previous local error')

    self.register_a_shortcut(
      'f', lambda: self.next_prev_marker_error(1, 'FP_list'),
      description='Find next false positive')
    self.register_a_shortcut(
      'd', lambda: self.next_prev_marker_error(-1, 'FP_list'),
      description='Find previous false positive')
  # endregion: Useful Commands

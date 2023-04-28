from freud.gui.sleep_monitor import SleepMonitor
from pictor.objects.signals.scrolling import Scrolling
from roma import console

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np



class CAMonitor(SleepMonitor):

  @SleepMonitor.property()
  def cam_buffer(self): return {}

  def stage(self,
            channels: str,
            t_file_path: str = None,
            model_name: str = None,
            stage_permutation: str = None,
            cam: bool = False,
            grad_cam: bool = False):
    """Auto-stage current PSG record using pretrained talos model"""
    from pictor.objects.signals.signal_group import SignalGroup, Annotation
    from freud.talos_utils.slp_config import SleepConfig
    from freud.talos_utils.slp_set import SleepSet

    import importlib.util

    if self._selected_signal is None:
      console.show_status(' ! No signal group found.')
      return

    # (0) Load module from t-file
    if t_file_path is None:
      t_file_path = self.pictor.load_file_dialog('Please select a t-file')
      if t_file_path == '': return
    module_name = 'this_name_does_not_matter'
    spec = importlib.util.spec_from_file_location(module_name, t_file_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    th: SleepConfig = mod.core.th
    th.developer_code += 'deactivate'

    # Execute main to load basic module settings
    mod.main(None)

    # (1) Prepare data set
    # Get displayed signal_group
    sg: SignalGroup = self._selected_signal
    ds = SleepSet(signal_groups=[sg])

    # Set CHANNELS for extracting tapes during configuration
    ds.CHANNELS = {f'{i}': k for i, k in enumerate(self.channel_list)}
    th.data_config = f'whatever {channels}'
    ds.configure()
    ds = ds.extract_data_set(include_targets=False)

    # (2) Prepare model
    from tframe import Classifier

    if model_name is None: model_name = mod.model_name
    model: Classifier = th.model()
    preds = model.classify(ds, batch_size=128, verbose=True)

    # (3) Set preds to annotations
    if stage_permutation is None: stage_permutation = '1,2,3,4,5'
    stage_map = {
      int(str_i) - 1: i for i, str_i in enumerate(stage_permutation.split(','))}

    t0 = sg.digital_signals[0].ticks[0]
    intervals = [(t0 + i * 30, t0 + (i + 1) * 30) for i, _ in enumerate(preds)]
    annotations = [stage_map[i] for i in preds]
    sg.annotations[f'stage {model_name}'] = Annotation(
      intervals, annotations, labels=SleepSet.AASM_LABELS)

    console.show_status(f'`{model_name}` set to sg.Annotation')

    self.toggle_annotation('stage', model_name)

    # TODO --------------------------------------------------------------------
    sg_heatmap = []
    # CAM
    if cam:
      dense_value = model.get_trainable_variables()['FeedforwardNet/dense/dense/dense/psi_1/W:0']
      last_conv = model.children[4].children[0].output_tensor
      last_conv_outputs = model.evaluate(last_conv, ds, 128)

      for index, last_conv_output in enumerate(last_conv_outputs):
        class_value = dense_value[:, preds[index]]
        heatmap = np.mean(last_conv_output * class_value, axis=1)
        heatmap = self.process_heatmap(heatmap)
        sg_heatmap.extend(heatmap)

    # Grad-CAM
    if grad_cam:
      from tframe import tf
      model = model
      last_conv = model.children[4].children[0].output_tensor
      dense = model.children[-1].children[0].output_tensor

      # get the predict result of every epoch [?, 5] -> [?, 1]
      dense = tf.reshape(tf.reduce_max(dense, reduction_indices=[1]), [-1, 1])

      # get last_conv_output and grads
      grads = tf.gradients(dense, last_conv)
      tensors = [last_conv, grads[0]]
      last_conv_outputs, grads_outputs = model.evaluate(tensors, ds, 128)

      for index, last_conv_output in enumerate(last_conv_outputs):
        feature_weight = np.mean(grads_outputs[index], axis=0)
        heatmap = np.mean(last_conv_output * feature_weight, axis=1)
        heatmap = self.process_heatmap(heatmap)
        sg_heatmap.extend(heatmap)

    self.cam_buffer['heatmap'] = sg_heatmap

  def process_heatmap(self, heatmap):
    # Convert heatmatp to [3000]
    from scipy.interpolate import interp1d
    x = np.linspace(0, heatmap.shape[0] - 1, heatmap.shape[0])
    func = interp1d(x, heatmap, kind='cubic')
    x_new = np.linspace(0, heatmap.shape[0] - 1, 3000)
    heatmap = func(x_new)
    # relu and normalize
    heatmap = np.array(np.maximum(heatmap, 0) / np.max(heatmap))
    return heatmap

  def _plot_curve(self, ax: plt.Axes, s: Scrolling):
    """ i  y
           2  ---------
        0     -> N(=2) - i(=0) - 0.5 = 1.5
           1  ---------
        1     -> N(=2) - i(=1) - 0.5 = 0.5
           0  ---------
    """
    # Get settings
    smart_scale = self.get('smart_scale')
    hl_id = self.get('hl')

    # Get channels [(name, x, y)]
    channels = s.get_channels(self.get('channels'),
                              max_ticks=self.get('max_ticks'))
    N = len(channels)

    margin = 0.05
    for i, (name, x, y) in enumerate(channels):
      # Normalized y before plot
      if not smart_scale:
        y = y - min(y)
        y = y / max(y) * (1.0 - 2 * margin) + margin
      else:
        xi = self.get('xi')
        mi = s.get_channel_percentile(name, xi)
        ma = s.get_channel_percentile(name, 100 - xi)
        y = y - mi
        y = y / (ma - mi) * (1.0 - 2 * margin) + margin

      y = y + N - 1 - i
      # Plot normalized y
      hm = self.cam_buffer['heatmap'] if self.cam_buffer.__contains__('heatmap') else []
      if len(hm) != 0 and i == 0:
        hm = self.cam_buffer['heatmap']
        start, end = (x[0] - 28830) * 100, (x[-1] - 28830) * 100
        hm = hm[int(start): int(end + 1)]
        if len(hm) != len(x):
          print('-' * 20)
          print('x_start:', x[0], 'x_end:', x[-1])
          print('h_start:', start, 'h_end:', end)
          print('-' * 20)
        ax.scatter(x, y, c=hm, cmap="OrRd", s=1)
      else:
        color, zorder = 'black', 10
        if 0 < hl_id != i + 1: color, zorder = '#AAA', None
        ax.plot(x, y, color=color, linewidth=1, zorder=zorder)

    # Set xlim (make sure display interval \in data interval)
    tick_list = [x for _, x, _ in channels]
    ax.set_xlim(max([x[0] for x in tick_list]),
                min([x[-1] for x in tick_list]))

    # Set y_ticks
    ax.set_yticks([N - i - 0.5 for i in range(N)])
    ax.set_yticklabels([name for name, _, _ in channels])

    # Highlight label if necessary
    if hl_id > 0:
      for i, label in enumerate(ax.get_yticklabels()):
        label.set_color('black' if i + 1 == hl_id else 'grey')

    # Set styles
    ax.set_ylim(0, N)
    ax.grid(color='#E03', alpha=0.4)

    tail = f' (xi={self.get("xi")})' if smart_scale else ''
    ax.set_title(s.label + tail)

  def show_curves(self, x: np.ndarray, fig: plt.Figure, i: int):

    super(CAMonitor, self).show_curves(x, fig, i)

    key = 'heatmap'
    if key not in self.cam_buffer: return
    # TODO --------------------------------------------------------------------
    # hm = self.cam_buffer['heatmap']
    #
    # # Find index
    # s: Scrolling = self._selected_signal
    # i1, i2 = None, None
    # hm = hm[i1:i2]
    #
    # f = None
    # im = f(hm)
    #
    # plt.imshow(im)
    #
    # # Show highlight

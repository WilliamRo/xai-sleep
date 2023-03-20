from freud.gui.sleep_monitor import SleepMonitor
from roma import console

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np



class CAMonitor(SleepMonitor):

  @SleepMonitor.property()
  def cam_buffer(self): return []

  def stage(self,
            channels: str,
            t_file_path: str = None,
            model_name: str = None,
            stage_permutation: str = None,
            cam: bool = False):
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

    print() # this is for creating a new line
    console.show_status(f'`{model_name}` set to sg.Annotation')

    self.toggle_annotation('stage', model_name)

    # TODO --------------------------------------------------------------------

    tensor = model.children[4].children[0].output_tensor
    tensors = [tensor]
    # results = model.evaluate(tensors, ds, 128)

    i = 0
    a = model.children[-1].children[0].output_tensor[:, i]

    from tframe import tf
    grads = tf.gradients(a[0], tensor)
    results = model.evaluate(grads[0], ds, 128)


  def show_curves(self, x: np.ndarray, fig: plt.Figure, i: int):
    super(CAMonitor, self).show_curves(x, fig, i)

    # TODO --------------------------------------------------------------------

    # Show highlight



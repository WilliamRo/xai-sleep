from freud.talos_utils.slp_config import SleepConfig
from freud.talos_utils.slp_set import SleepSet
from pictor.objects.signals.signal_group import Annotation, SignalGroup

import importlib.util



def stage_alpha(sg: SignalGroup, t_file_path: str) -> Annotation:
  # (0) Load module from t-file
  # Load model
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
  ds = SleepSet(signal_groups=[sg])

  # Set CHANNELS for extracting tapes during configuration
  # TODO: channels
  # channels = '1,2'
  # ds.CHANNELS = {f'{i}': k for i, k in enumerate(self.channel_list)}
  # th.data_config = f'whatever {channels}'

  ds.configure()
  ds = ds.extract_data_set(include_targets=False)

  # (2) Prepare model
  from tframe import Classifier

  model: Classifier = th.model()
  preds = model.classify(ds, batch_size=128, verbose=True)

  # (3) Set preds to annotations
  stage_permutation = '1,2,3,4,5'
  stage_map = {
    int(str_i) - 1: i for i, str_i in enumerate(stage_permutation.split(','))}

  t0 = sg.digital_signals[0].ticks[0]
  intervals = [(t0 + i * 30, t0 + (i + 1) * 30) for i, _ in enumerate(preds)]
  annotations = [stage_map[i] for i in preds]
  anno = Annotation(intervals, annotations, labels=SleepSet.AASM_LABELS)
  return anno


def compare(gt_anno: Annotation, pred_anno: Annotation):
  cm = None
  return cm

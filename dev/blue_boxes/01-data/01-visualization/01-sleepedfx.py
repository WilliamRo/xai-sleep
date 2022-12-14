from freud.talos_utils.sleep_sets.sleepedfx import SleepEDFx
from pictor.objects.signals.signal_group import SignalGroup, Annotation



data_root = r'../../../../data/sleepedf'

ds = SleepEDFx.load_as_sleep_set(data_root)

sg0: SignalGroup = ds.signal_groups[0]
intervals, annotations = [(0, 70000)], [0]
sg0.annotations['stage Predicted'] = Annotation(
  intervals, annotations, labels=SleepEDFx.ANNO_LABELS)

ds.show()


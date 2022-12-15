from freud.talos_utils.sleep_sets.ucddb import UCDDB
from pictor.objects.signals.signal_group import SignalGroup, Annotation



data_root = r'../../../../data/ucddb'

ds = UCDDB.load_as_sleep_set(data_root)

sg0: SignalGroup = ds.signal_groups[0]
intervals, annotations = [(0, 70000)], [0]
sg0.annotations['stage Predicted'] = Annotation(
  intervals, annotations, labels=UCDDB.ANNO_LABELS)

def config(m):
  from pictor.plotters.monitor import Monitor
  assert isinstance(m, Monitor)
  m.toggle_annotation(auto_refresh=False)
  m.toggle_annotation(anno_label='Predicted', auto_refresh=False)

ds.show(config)


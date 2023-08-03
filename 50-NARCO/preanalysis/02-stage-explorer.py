from freud.talos_utils.sleep_sets.rrshv1 import RRSHSCv1
from freud.gui.freud_gui import Freud
from tframe import console



console.suppress_logging()

data_dirs = [
  r'../../data/rrsh',
  r'../../data/rrsh-narcolepsy',
]

signal_groups = []
for path in data_dirs:
  signal_groups.extend(RRSHSCv1.load_as_signal_groups(path))


freud = Freud('Nacrolepsy Explorer')
freud.objects = signal_groups
freud.monitor.set('channels', 'PositionSen', auto_refresh=False)
freud.monitor.set('default_win_duration', 50000, auto_refresh=False)
freud.monitor._annotations_to_show = ['stage Ground-Truth']

freud.show()


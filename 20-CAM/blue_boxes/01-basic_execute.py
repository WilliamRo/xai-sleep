from freud.talos_utils.sleep_sets.sleepedfx import SleepEDFx, SleepSet
from tframe import console
from cam_oliver.oliver import Oliver



console.suppress_logging()
data_dir = r'../../data/sleepedfx'
signal_groups = SleepEDFx.load_as_signal_groups(data_dir)

oliver = Oliver(title='Oliver')
oliver.objects = signal_groups
oliver.show()

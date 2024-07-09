from freud.gui.freud_gui import Freud
from pictor.objects.signals import SignalGroup
from roma import finder, io



# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
data_dir = r'../../../data/sleep-edf-database-expanded-1.0.0/sleep-cassette/'
pattern = f'*(raw).sg'

N = 6
# -----------------------------------------------------------------------------
# Read data
# -----------------------------------------------------------------------------
sg_file_list = finder.walk(data_dir, pattern=pattern)[:N]

signal_groups = []
for path in sg_file_list[:N]:
  sg: SignalGroup = io.load_file(path, verbose=True)
  signal_groups.append(sg)

# -----------------------------------------------------------------------------
# Visualize data
# -----------------------------------------------------------------------------
Freud.visualize_signal_groups(
  signal_groups,
  title='SleepEDFx-SC-Raw-SG',
  default_win_duration=9999999,
)



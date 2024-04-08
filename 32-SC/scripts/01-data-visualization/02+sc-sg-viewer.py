from roma import finder
from roma import io
from pictor.objects.signals.signal_group import SignalGroup



# Configs
N = 20

# Select .sg files
data_dir = r'../../../data/sleep-edf-database-expanded-1.0.0/sleep-cassette/'
pattern = f'*(trim1800;128).sg'

sg_file_list = finder.walk(data_dir, pattern=pattern)[:N]

signal_groups = []
for path in sg_file_list[:N]:
  sg: SignalGroup = io.load_file(path, verbose=True)
  signal_groups.append(sg)

# Show signal groups in Freud
from freud.gui.freud_gui import Freud

# Initialize pictor and set objects
Freud.visualize_signal_groups(signal_groups, 'SleepEDFx-SC',
                              default_win_duration=9999999)

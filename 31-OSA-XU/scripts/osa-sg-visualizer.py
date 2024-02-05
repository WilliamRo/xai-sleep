from roma import finder
from roma import io
from pictor.objects.signals.signal_group import SignalGroup, Annotation



# Configs
N = 10

# Select .sg files
data_dir = r'../../data/rrsh-osa'
pattern = f'*(trim;simple;100).sg'

sg_file_list = finder.walk(data_dir, pattern=pattern)[:N]

signal_groups = []
for path in sg_file_list:
  sg: SignalGroup = io.load_file(path, verbose=True)
  signal_groups.append(sg)

# Show signal groups in Freud
from freud.gui.freud_gui import Freud

# Initialize pictor and set objects
Freud.visualize_signal_groups(signal_groups, 'RRSH-OSA',
                              default_win_duration=9999999)

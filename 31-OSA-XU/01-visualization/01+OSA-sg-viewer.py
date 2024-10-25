from roma import finder
from roma import io
from pictor.objects.signals.signal_group import SignalGroup



# Configs
N = 20

# Select .sg files
SG_DIR = r'../../data/rrsh-osa'
SG_PATTERN = f'*(trim;simple;100).sg'
SG_PATTERN = f'111(trim;simple;100).sg'


sg_file_list = finder.walk(SG_DIR, pattern=SG_PATTERN)[:N]

signal_groups = []
for path in sg_file_list[:N]:
  sg: SignalGroup = io.load_file(path, verbose=True)
  signal_groups.append(sg)

# Show signal groups in Freud
from freud.gui.freud_gui import Freud

# Initialize pictor and set objects
Freud.visualize_signal_groups(signal_groups, 'OSA-XU',
                              default_win_duration=9999999)

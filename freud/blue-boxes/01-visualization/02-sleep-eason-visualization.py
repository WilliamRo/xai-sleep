from freud.gui.freud_gui import Freud
from freud.talos_utils.sleep_sets.sleepeason import SleepEason
from fnmatch import fnmatch
from pictor.objects import SignalGroup
from roma import finder
from roma import io



# Set directories
data_dir = r'../../../data/'
data_dir += 'sleepeason1'

prefix = ['', 'sleepedfx', 'ucddb', 'rrsh'][2]
pattern = f'{prefix}*.sg'

# Select .sg files
sg_file_list = finder.walk(data_dir, pattern=pattern)

signal_groups = []
for path in sg_file_list:
  sg = io.load_file(path, verbose=True)
  signal_groups.append(sg)

# Visualize signal groups
Freud.visualize_signal_groups(signal_groups, figure_size=(11, 6))






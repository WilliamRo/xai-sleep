from pictor.objects.signals.signal_group import SignalGroup, Annotation
from roma import finder
from roma import io

from epoch_explorer_base import EpochExplorer
from epoch_explorer_omic import RhythmPlotterPro
from ee_walker import RhythmWalker



# Set directories
data_dir = r'../../../data/'
data_dir += 'rrsh-osa'


pattern = f'*(trim;easy;100).sg'
channel_names = ['E1-M2', 'E2-M2']

# Select .sg files
sg_file_list = finder.walk(data_dir, pattern=pattern)

signal_groups = []
for path in sg_file_list:
  sg: SignalGroup = io.load_file(path, verbose=True)
  if channel_names: sg = sg.extract_channels(channel_names)
  signal_groups.append(sg)

# Visualize signal groups
EpochExplorer.explore(signal_groups, plot_wave=True,
                      plotter_cls=RhythmWalker)


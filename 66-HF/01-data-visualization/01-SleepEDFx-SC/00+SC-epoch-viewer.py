from ee_walker import EpochExplorer, RhythmWalker, SignalGroup
from roma import finder
from roma import io



# Configurations
# (1) configure raw data
DATA_DIR = r'../../../data/sleep-edf-database-expanded-1.0.0/sleep-cassette'
PATTERN = f'*(trim1800;128).sg'
CHANNELS = ['EEG Fpz-Cz', 'EEG Pz-Oz']

N = 10

# Select .sg files
sg_file_list = finder.walk(DATA_DIR, pattern=PATTERN)
sg_file_list = sg_file_list[:N]

signal_groups = []
for path in sg_file_list:
  sg: SignalGroup = io.load_file(path, verbose=True)
  sg = sg.extract_channels(CHANNELS)
  signal_groups.append(sg)

# Visualize signal groups
ee = EpochExplorer.explore(signal_groups, plot_wave=True)


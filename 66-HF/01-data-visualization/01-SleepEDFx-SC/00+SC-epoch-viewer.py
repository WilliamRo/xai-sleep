from ee_walker import EpochExplorer, SignalGroup
from roma import finder
from roma import io

import sc as hub



# Configurations
# (1) configure raw data
PATTERN = f'*(trim1800;128).sg'
CHANNELS = ['EEG Fpz-Cz', 'EEG Pz-Oz']

N = 2

# Select .sg files
sg_file_list = finder.walk(hub.SG_DIR, pattern=PATTERN)
sg_file_list = sg_file_list[:N]

signal_groups = []
for path in sg_file_list:
  sg: SignalGroup = io.load_file(path, verbose=True)
  sg = sg.extract_channels(CHANNELS)
  signal_groups.append(sg)

# Visualize signal groups
ee = EpochExplorer.explore(signal_groups, plot_wave=True)


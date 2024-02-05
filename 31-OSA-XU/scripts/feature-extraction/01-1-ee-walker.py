from ee_walker import EpochExplorer, RhythmWalker, SignalGroup
from roma import finder
from roma import io

import os



# Configurations
# (1) configure raw data
DATA_DIR = r'../../../data/rrsh-osa'
PATTERN = f'*(trim;simple;100).sg'

# (2) configure PSG fingerprint extractor
FEATURE_DIR = r'P:\xai-sleep\31-OSA-XU\features'
SAVE_FILE_NAME = '0203-N125-ch(all)-F(20)-A(128).fp'
CHANNELS = [
  'E1-M2', 'E2-M2',
  'F3-M2', 'C3-M2',
  'O1-M2', 'F4-M1',
  'C4-M1', 'O2-M1',
]
ARGS = {'BM01-FREQ': ('max_freq', [20]),
        'BM02-AMP': ('pool_size', [128])}


class OSAWalker(RhythmWalker):
  def export_osa_features(self):
    sg_list = self.explorer.axes[self.explorer.Keys.OBJECTS]
    biomarkers = {'BM01-FREQ': self._bm_01_freq, 'BM02-AMP': self._bm_02_ampl}
    self.walk_on_selected_sg(sg_list, CHANNELS, biomarkers, ARGS,
                             os.path.join(FEATURE_DIR, SAVE_FILE_NAME))
  eof = export_osa_features



# Select .sg files
sg_file_list = finder.walk(DATA_DIR, pattern=PATTERN)
sg_file_list = sorted(
  sg_file_list, key=lambda fn: int(fn.split('/')[5].split('(')[0]))

signal_groups = []
for path in sg_file_list:
  sg: SignalGroup = io.load_file(path, verbose=True)
  sg = sg.extract_channels(CHANNELS)
  signal_groups.append(sg)

# Visualize signal groups
EpochExplorer.explore(signal_groups, plot_wave=True, plotter_cls=OSAWalker)

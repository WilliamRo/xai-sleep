from ee_walker import EpochExplorer, RhythmWalker, SignalGroup
from roma import finder
from roma import io

import os



# Configurations
# (1) configure raw data
DATA_DIR = r'../../../data/sleep-edf-database-expanded-1.0.0/sleep-cassette'
PATTERN = f'*(trim1800;128).sg'

# (2) configure PSG fingerprint extractor
FEATURE_DIR = r'P:\xai-sleep\32-SC\features'
SAVE_FILE_NAME = '0211-N153-ch(all)-F(20)-A(128).fp'
CHANNELS = ['EEG Fpz-Cz', 'EEG Pz-Oz']
ARGS = {'BM01-FREQ': ('max_freq', [20, 30]),
        'BM02-AMP': ('pool_size', [128])}


class SCWalker(RhythmWalker):
  def export_sc_features(self):
    sg_list = self.explorer.axes[self.explorer.Keys.OBJECTS]
    biomarkers = {'BM01-FREQ': self._bm_01_freq, 'BM02-AMP': self._bm_02_ampl}
    self.walk_on_selected_sg(sg_list, CHANNELS, biomarkers, ARGS,
                             os.path.join(FEATURE_DIR, SAVE_FILE_NAME))


# Select .sg files
sg_file_list = finder.walk(DATA_DIR, pattern=PATTERN)

signal_groups = []
for path in sg_file_list:
  sg: SignalGroup = io.load_file(path, verbose=True)
  sg = sg.extract_channels(CHANNELS)
  signal_groups.append(sg)

# Visualize signal groups
ee = EpochExplorer.explore(
  signal_groups, plot_wave=True, plotter_cls=SCWalker, dont_show=True)
ee.rhythm_plotter.export_sc_features()


import numpy as np

from pictor import Pictor

import os
import matplotlib.pyplot as plt



# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
# (1) Specify edf and hypnogram file paths
data_dir = r'../../../data/sleep-edf-database-expanded-1.0.0/sleep-cassette/'
edf_file_name = r'SC4001E0-PSG.edf'
hypnogram_file_name = r'SC4001EC-Hypnogram.edf'

edf_file_path = os.path.join(data_dir, edf_file_name)
hypnogram_file_path = os.path.join(data_dir, hypnogram_file_name)

assert os.path.exists(edf_file_path), f'Path not found: {edf_file_path}'
assert os.path.exists(hypnogram_file_path), f'Path not found: {hypnogram_file_path}'

# (2) Specify channel for visualization
channel_name = 'EEG Fpz-Cz'

# -----------------------------------------------------------------------------
# Read data as numpy array, shape = (T * sfreq, )
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Visualize data in pictor
# -----------------------------------------------------------------------------

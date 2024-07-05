from pictor import Pictor

import matplotlib.pyplot as plt
import numpy as np
import os



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
import mne

# (1) Read PSG data
with mne.io.read_raw_edf(
    edf_file_path, include=[channel_name], preload=False) as file:

  sampling_frequency = file.info['sfreq']
  data = file.get_data()[0]

# (2) Read annotation
mne_anno: mne.Annotations = mne.read_annotations(hypnogram_file_path)
sleep_stages = []
print(f'Total duration = {sum(mne_anno.duration)}')
for duration, description in zip(mne_anno.duration, mne_anno.description):
  sleep_stages.extend([description] * int(duration / 30))

# -----------------------------------------------------------------------------
# Visualize data in pictor
# -----------------------------------------------------------------------------
ticks_per_epoch = int(30 * sampling_frequency)
n_epochs = len(data) // ticks_per_epoch
data = np.reshape(data[:n_epochs * ticks_per_epoch],
                  (n_epochs, ticks_per_epoch))

# assert len(sleep_stages) == len(data), f'{len(sleep_stages)} != {len(data)}'
sleep_stages = sleep_stages[:n_epochs]

def plot_epoch(x: np.ndarray, label: str, ax: plt.Axes):
  t = np.arange(len(x)) / sampling_frequency
  ax.plot(t, x * 1e6)
  ax.grid(True)

  ax.set_title(label)
  ax.set_xlabel('Time (s)')
  ax.set_ylabel('Amplitude (uV)')

p = Pictor(channel_name, figure_size=(8, 4))
p.objects = data
p.labels = sleep_stages
p.add_plotter(plot_epoch)
p.show()

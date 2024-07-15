from freud.gui.freud_gui import Freud
from pictor.objects.signals import SignalGroup, Annotation, DigitalSignal

import mne
import os



# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
# (1) Specify edf and hypnogram file paths
data_dir = r'../../../data/sleep-edf-database-expanded-1.0.0/sleep-cassette/'
data_files = [(r'SC4001E0-PSG.edf', r'SC4001EC-Hypnogram.edf'),
              (r'SC4002E0-PSG.edf', r'SC4002EC-Hypnogram.edf')]

# -----------------------------------------------------------------------------
# Read data and wrap data into SignalGroups
# -----------------------------------------------------------------------------
signal_groups = []
map_dict = {
  'Sleep stage W': 0,
  'Sleep stage 1': 1,
  'Sleep stage 2': 2,
  'Sleep stage 3': 3,
  'Sleep stage 4': 4,
  'Sleep stage R': 5,
}
anno_labels = ['W', 'N1', 'N2', 'N3', 'N4', 'REM', '?']

for psg_fn, ann_fn in data_files:
  edf_file_path = os.path.join(data_dir, psg_fn)
  hypnogram_file_path = os.path.join(data_dir, ann_fn)

  # (1) Read PSG data
  with mne.io.read_raw_edf(edf_file_path, preload=False) as file:
    # -------------------------------------------------------------------------
    # Begin
    # -------------------------------------------------------------------------
    # data.shape = (n_ticks, n_channels)
    data = None
    # sampling_frequency = XXX Hz
    sampling_frequency = None
    ds = DigitalSignal(data, sampling_frequency, channel_names=file.ch_names)
    sg = SignalGroup(ds, label=psg_fn.split('-')[0])

    # intervals = [(start_1, end_1), (start_2, end_2), ...]
    intervals = None
    # annotations = [0, 1, 2, 1, 3, ...]
    annotations = None
    anno = Annotation(intervals, annotations, anno_labels)
    # -------------------------------------------------------------------------
    # End
    # -------------------------------------------------------------------------
    sg.annotations['stage Ground-Truth'] = anno
    signal_groups.append(sg)

# -----------------------------------------------------------------------------
# Visualize data
# -----------------------------------------------------------------------------
Freud.visualize_signal_groups(
  signal_groups,
  title='SleepEDFx-SC',
  default_win_duration=9999999,
)



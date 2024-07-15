from freud.gui.freud_gui import Freud
from pictor.objects.signals import SignalGroup, Annotation, DigitalSignal
from roma import io

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

  # Get PID, e.g., SC4001E0-PSG.edf -> SC4001E
  pid = psg_fn.split('-')[0][:-1]
  sg_file_name = f'{pid}(raw).sg'
  sg_file_path = os.path.join(data_dir, sg_file_name)

  if os.path.exists(sg_file_path):
    sg: SignalGroup = io.load_file(sg_file_path, verbose=True)
    signal_groups.append(sg)
    continue

  # (1) Read PSG data
  with mne.io.read_raw_edf(edf_file_path, preload=False) as file:
    sampling_frequency = file.info['sfreq']
    data = file.get_data()

    ds = DigitalSignal(data.T, sampling_frequency, channel_names=file.ch_names)
    sg = SignalGroup(ds, label=pid)

    # (2) Read annotation
    mne_anno: mne.Annotations = mne.read_annotations(hypnogram_file_path)
    intervals = []
    annotations = []
    for onset, duration, description in zip(
        mne_anno.onset, mne_anno.duration, mne_anno.description):
      intervals.append((onset, onset + duration))

      if description in map_dict: value = map_dict[description]
      else: value = 6
      annotations.append(value)

    anno = Annotation(intervals, annotations, anno_labels)
    sg.annotations['stage Ground-Truth'] = anno

    # (3) Put sg into bucket
    signal_groups.append(sg)

    # (4) Save .sg file if not exist
    io.save_file(sg, sg_file_path, verbose=True)

# -----------------------------------------------------------------------------
# Visualize data
# -----------------------------------------------------------------------------
Freud.visualize_signal_groups(
  signal_groups,
  title='SleepEDFx-SC',
  default_win_duration=9999999,
)



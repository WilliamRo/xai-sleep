from pictor.objects.signals.digital_signal import DigitalSignal
from pictor.objects.signals.signal_group import Annotation
from typing import List

import os
import numpy as np



def read_digital_signals_mne(
    file_path: str,
    groups=None,
    dtype=np.float32,
    **kwargs
) -> List[DigitalSignal]:
  """Read .edf file using `mne` package.

  :param groups: A list/tuple of channel names groups by sampling frequency.
         If not provided, data will be read in a channel by channel fashion.
  :param max_sfreq: maximum sampling frequency
  :param allow_rename: option to allow rename file when target extension is
         not .edf.
  """
  import mne.io

  max_sfreq = kwargs.get('max_sfreq', None)
  chn_map = kwargs.get('chn_map', None)

  # Rename file if necessary
  if file_path[-4:] != '.edf' and kwargs.get('allow_rename', False):
    os.rename(file_path, file_path + '.edf')
    file_path += '.edf'

  open_file = lambda exclude=(): mne.io.read_raw_edf(
    file_path, exclude=exclude, preload=False, verbose=False)

  # Read all channel names, create a reverse map
  with open_file() as file: edf_channel_names = file.ch_names
  if callable(chn_map):
    chn_map_dict = {chn: chn_map(chn) for chn in edf_channel_names}
  else:
    chn_map_dict = {chn: chn for chn in edf_channel_names}

  chn_rev_map = {v: k for k, v in chn_map_dict.items()}

  # Initialize groups if not provided, otherwise get channel_names from groups
  if groups is None:
    channel_names = edf_channel_names
    groups = [[chn] for chn in channel_names]
  else:
    channel_names = [chn for g in groups for chn in g]

  # Generate exclude lists
  exclude_lists = [[chn_rev_map[chn] for chn in channel_names if chn not in g]
                   for g in groups]

  # Read raw data
  signal_dict = {}
  for exclude_list in exclude_lists:
    with open_file(exclude_list) as file:
      sfreq = file.info['sfreq']

      # Resample to `max_sfreq` if necessary
      if max_sfreq is not None and sfreq > max_sfreq:
        file.resample(max_sfreq)
        sfreq = max_sfreq

      # Read signal
      if sfreq not in signal_dict: signal_dict[sfreq] = []
      signal_dict[sfreq].append((file.ch_names, file.get_data()))

  # Wrap data into DigitalSignals
  digital_signals = []
  for sfreq, signal_lists in signal_dict.items():
    data = np.concatenate([x for _, x in signal_lists], axis=0)
    data = np.transpose(data).astype(dtype)

    channel_names = [chn_map_dict[name] for names, _ in signal_lists
                     for name in names]

    digital_signals.append(DigitalSignal(
      data, channel_names=channel_names, sfreq=sfreq,
      label=','.join(channel_names)))

  return digital_signals


def read_annotations_mne(file_path: str, labels=None) -> Annotation:
  """Read annotations using `mne` package"""
  import mne

  # Read mne.Annotations
  mne_anno: mne.Annotations = mne.read_annotations(file_path)

  # Automatically generate labels if necessary
  if labels is None: labels = list(sorted(set(mne_anno.description)))

  # Read intervals and annotations
  intervals, annotations = [], []
  label2int = {lb: i for i, lb in enumerate(labels)}
  for onset, duration, label in zip(
      mne_anno.onset, mne_anno.duration, mne_anno.description):
    intervals.append((onset, onset + duration))
    annotations.append(label2int[label])

  return Annotation(intervals, annotations, labels=labels)

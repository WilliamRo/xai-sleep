from collections import OrderedDict
from hypnomics.hypnoprints import extract_hypnocloud_from_signal_group
from hypnomics.hypnoprints.hp_extractor import extract_hypnoprints_from_hypnocloud
from roma import finder
from roma import io
from pictor.objects.signals.signal_group import SignalGroup

from sc.fp_viewer import FPViewer

import numpy as np
import os



# -----------------------------------------------------------------------------
# Configs
# -----------------------------------------------------------------------------
# Select .sg files
feature_dir = r'../../../33-ST/features/'
time_reso = [30, 15, 10, 6, 2][0]
file_name = f'C2-dt{time_reso}.clouds'

# 0: placebo, 1: Temazepam
targets = np.load(os.path.join(feature_dir, 'targets.npz'))['y']


clouds = io.load_file(os.path.join(feature_dir, file_name), verbose=True)
cloud_dict = OrderedDict()
for i, cloud in enumerate(clouds):

  # if i // 2 + 1 in (3, 4, 5, 14, 15, 17, 18, 19, 21, 22): continue

  cloud_key = f'ST-{i // 2 + 1:02d}-'
  if targets[i] == 0: cloud_key += 'P'
  else: cloud_key += 'T'
  cloud_dict[cloud_key] = cloud


# Extract hypnoprints
channels = ['EEG Fpz-Cz', 'EEG Pz-Oz']

# -----------------------------------------------------------------------------
# Make an FPS
# -----------------------------------------------------------------------------
STAGE_KEYS = ('W', 'N1', 'N2', 'N3', 'R')

fps = {}
fps['meta'] = (list(cloud_dict.keys()), channels,
               {'FREQ': ('max_freq', [20]), 'AMP': ('pool_size', [128])})
for cloud_key, cloud in cloud_dict.items():
  for chn in channels:
    for pk in ('amplitude', 'frequency'):
      bk = {'amplitude': ('AMP', 'pool_size', 128),
            'frequency': ('FREQ', 'max_freq', 20)}[pk]
      key = (cloud_key, chn, bk)
      fps[key] = {}
      for sk in STAGE_KEYS:
        data = cloud[chn][sk][pk]
        if 'amp' in pk: data = data / 1e9
        fps[key][sk] = np.array(data, dtype=np.float32)

fpv = FPViewer(walker_results=fps)
fpv.show()



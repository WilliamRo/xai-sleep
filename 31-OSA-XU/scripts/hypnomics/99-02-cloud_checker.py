from hypnomics.hypnoprints import extract_hypnocloud_from_signal_group
from hypnomics.hypnoprints.hp_extractor import extract_hypnoprints_from_hypnocloud
from roma import finder
from roma import io, console
from pictor.objects.signals.signal_group import SignalGroup

from sc.fp_viewer import FPViewer
from collections import OrderedDict

import os



# Configs
N = 999
# N = 5
reso = 30

# Select .sg files
data_dir = r'../../../data/rrsh-osa/'
pattern = f'*(trim;simple;100).sg'

sg_file_list = finder.walk(data_dir, pattern=pattern)[:N]
N = len(sg_file_list)

signal_groups = []
for path in sg_file_list[:N]:
  sg: SignalGroup = io.load_file(path, verbose=True)
  signal_groups.append(sg)

# Extract hypnoprints
channels = [
  # 'E1-M2',
  # 'E2-M2',
  'F3-M2',
  # 'F4-M1',
  'C3-M2',
  # 'C4-M1',
  # 'O1-M2',
  # 'O2-M1',
]
NC = len(channels)

# Save cloud files
cloud_data_dir = r'../../features/'
cloud_file_name = f'OSA-{N}pts-{NC}chs-{reso}s.clouds'
file_path = os.path.join(cloud_data_dir, cloud_file_name)
clouds = io.load_file(file_path, verbose=True)

if isinstance(clouds, list):
  cloud_dict = OrderedDict()
  for sg, cloud in zip(signal_groups, clouds):
    cloud_dict[sg.label] = cloud
  io.save_file(cloud_dict, file_path, verbose=True)
else: cloud_dict = clouds

# -----------------------------------------------------------------------------
# Make an FPS
# -----------------------------------------------------------------------------
STAGE_KEYS = ('W', 'N1', 'N2', 'N3', 'R')

fps = {}
fps['meta'] = ([sg.label for sg in signal_groups], channels,
               {'FREQ': ('max_freq', [20]), 'AMP': ('pool_size', [128])})
for label, cloud in cloud_dict.items():
  for chn in channels:
    for pk in ('amplitude', 'frequency'):
      bk = {'amplitude': ('AMP', 'pool_size', 128),
            'frequency': ('FREQ', 'max_freq', 20)}[pk]
      key = (label, chn, bk)
      fps[key] = {}
      for sk in STAGE_KEYS: fps[key][sk] = cloud[chn][sk][pk]

fpv = FPViewer(walker_results=fps)
fpv.show()



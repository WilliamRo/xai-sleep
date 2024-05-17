from hypnomics.hypnoprints import extract_hypnocloud_from_signal_group
from hypnomics.hypnoprints.hp_extractor import extract_hypnoprints_from_hypnocloud
from roma import finder
from roma import io, console
from pictor.objects.signals.signal_group import SignalGroup

from sc.fp_viewer import FPViewer

from collections import OrderedDict

import os



# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
# Configs
N = 999
R = 30
overwrite = False

data_dir = r'../../../data/sleep-edf-database-expanded-1.0.0/sleep-cassette/'
pattern = f'*(trim1800;128).sg'

N = min(N, 153)

save_to_dir = r'../../features/'
cloud_file_name = f'SC-pt{N}-C2-dt{R}.clouds'
save_path = os.path.join(save_to_dir, cloud_file_name)

channels = ['EEG Fpz-Cz', 'EEG Pz-Oz']
# -----------------------------------------------------------------------------
# Read .sg files
# -----------------------------------------------------------------------------
if not os.path.exists(save_path) or overwrite:
  sg_file_list = finder.walk(data_dir, pattern=pattern)[:N]
  N = len(sg_file_list)

  signal_groups = []
  for path in sg_file_list:
    sg: SignalGroup = io.load_file(path, verbose=True)
    signal_groups.append(sg)

# -----------------------------------------------------------------------------
# Extract clouds and save
# -----------------------------------------------------------------------------
  # Extract hypnoprints
  clouds = OrderedDict()

  console.show_status(f'Converting {N} sg files to clouds ...')
  for i, sg in enumerate(signal_groups):
    console.print_progress(i, N)
    clouds[sg.label] = extract_hypnocloud_from_signal_group(
      sg, channels, time_resolution=R)
  console.show_status(f'Converted {N} sg files.')

  io.save_file(clouds, os.path.join(save_to_dir, cloud_file_name), verbose=True)
else:
  clouds = io.load_file(save_path, verbose=True)

# -----------------------------------------------------------------------------
# Make an FPS
# -----------------------------------------------------------------------------
STAGE_KEYS = ('W', 'N1', 'N2', 'N3', 'R')

fps = {}
fps['meta'] = (list(clouds.keys()), channels,
               {'FREQ': ('max_freq', [20]), 'AMP': ('pool_size', [128])})
for label, cloud in clouds.items():
  for chn in channels:
    for pk in ('amplitude', 'frequency'):
      bk = {'amplitude': ('AMP', 'pool_size', 128),
            'frequency': ('FREQ', 'max_freq', 20)}[pk]
      key = (label, chn, bk)
      fps[key] = {}
      for sk in STAGE_KEYS: fps[key][sk] = cloud[chn][sk][pk]

fpv = FPViewer(walker_results=fps)
fpv.show()



from hypnomics.hypnoprints import extract_hypnocloud_from_signal_group
from hypnomics.hypnoprints.hp_extractor import extract_hypnoprints_from_hypnocloud
from roma import finder
from roma import io
from pictor.objects.signals.signal_group import SignalGroup

from sc.fp_viewer import FPViewer



# Configs
N = 2

# Select .sg files
data_dir = r'../../../data/sleep-edf-database-expanded-1.0.0/sleep-cassette/'
pattern = f'*(trim1800;128).sg'

sg_file_list = finder.walk(data_dir, pattern=pattern)[:N]

signal_groups = []
for path in sg_file_list[:N]:
  sg: SignalGroup = io.load_file(path, verbose=True)
  signal_groups.append(sg)


# Extract hypnoprints
channels = ['EEG Fpz-Cz', 'EEG Pz-Oz']

clouds = [extract_hypnocloud_from_signal_group(sg, channels, time_resolution=6)
          for sg in signal_groups]

# -----------------------------------------------------------------------------
# Make an FPS
# -----------------------------------------------------------------------------
STAGE_KEYS = ('W', 'N1', 'N2', 'N3', 'R')

fps = {}
fps['meta'] = ([sg.label for sg in signal_groups], channels,
               {'FREQ': ('max_freq', [20]), 'AMP': ('pool_size', [128])})
for sg, cloud in zip(signal_groups, clouds):
  for chn in channels:
    for pk in ('amplitude', 'frequency'):
      bk = {'amplitude': ('AMP', 'pool_size', 128),
            'frequency': ('FREQ', 'max_freq', 20)}[pk]
      key = (sg.label, chn, bk)
      fps[key] = {}
      for sk in STAGE_KEYS: fps[key][sk] = cloud[chn][sk][pk]

fpv = FPViewer(walker_results=fps)
fpv.show()



from hypnomics.hypnoprints import extract_hypnocloud_from_signal_group
from hypnomics.hypnoprints import extract_hypnoprints_from_hypnocloud
from roma import finder
from roma import io
from pictor.objects.signals.signal_group import SignalGroup



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


# Extract clouds
channels = ['EEG Fpz-Cz', 'EEG Pz-Oz']
time_resolution = 30

clouds = [extract_hypnocloud_from_signal_group(
  sg, channels, time_resolution=time_resolution)
  for sg in signal_groups]

# Save if necessary
# io.save_file(clouds, f'C2-dt{time_resolution}.clouds')

# Generate feature
features = [extract_hypnoprints_from_hypnocloud(c) for c in clouds]



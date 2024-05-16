from hypnomics.hypnoprints import extract_hypnocloud_from_signal_group
from hypnomics.hypnoprints.hp_extractor import extract_hypnoprints_from_hypnocloud
from roma import finder
from roma import io, console
from pictor.objects.signals.signal_group import SignalGroup

from sc.fp_viewer import FPViewer

import os



# Configs
N = 125
reso = 30
NC = 2

# Save cloud files
cloud_data_dir = r'../../features/'
cloud_file_name = f'OSA-{N}pts-{NC}chs-{reso}s.clouds'
clouds = io.load_file(os.path.join(cloud_data_dir, cloud_file_name), verbose=True)

print()


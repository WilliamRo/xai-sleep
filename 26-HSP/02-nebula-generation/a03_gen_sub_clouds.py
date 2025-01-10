# Add path in order to be compatible with Linux
import sys, os

SOLUTION_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
print(f'Solution dir = {SOLUTION_DIR}')

PATH_LIST = ['26-HSP', '26-HSP/01-sg-conversion', 'xai-kit', 'hypnomics',
             '26-HSP/99-data-probe', 'xai-kit/roma', 'xai-kit/pictor',
             'xai-kit/tframe']

sys.path.append(SOLUTION_DIR)
for p in PATH_LIST: sys.path.append(os.path.join(SOLUTION_DIR, p))

# Import anything here
from freud.hypno_tools.probe_tools import get_probe_keys, get_extractor_dict
from hypnomics.freud.freud import Freud
from roma.console.console import console

import a00_common as hub
import numpy as np



# -----------------------------------------------------------------------------
# (1) Configuration
# -----------------------------------------------------------------------------
OVERWRITE = 0
SUBSET_FN = hub.SubsetDicts.ss_2ses_3types_378

CHANNELS = ['EEG F3-M2', 'EEG C3-M2', 'EEG O1-M2',
            'EEG F4-M1', 'EEG C4-M1', 'EEG O2-M1']
TIME_RESOLUTIONS = [30]

PROBE_CONFIG = 'c'
if len(sys.argv) > 1:
  assert len(sys.argv) == 2
  PROBE_CONFIG = sys.argv[1]

EXTRACTOR_KEYS = get_probe_keys(PROBE_CONFIG)
EXTRACTOR_KEYS = ['KURT']
extractor_dict = get_extractor_dict(EXTRACTOR_KEYS, fs=128)

console.show_status(f'extractor_keys = {list(extractor_dict.keys())}')
# -----------------------------------------------------------------------------
# (2) Cloud generation
# -----------------------------------------------------------------------------
freud = Freud(hub.CLOUD_DIR)

ho_list = hub.ha.load_subset_dict(SUBSET_FN, return_ho=1)
sg_file_list = [
  os.path.join(hub.SG_DIR, ho.get_sg_file_name(dtype=np.float16, max_sfreq=128))
  for ho in ho_list]

SG_PATTERN = r'sub-S*_ses-?(float16,128Hz).sg'
fs = freud.get_sampling_frequency(hub.SG_DIR, SG_PATTERN, CHANNELS)
assert fs == 128

console.show_status(f'Converting (PROBE_CONFIG={PROBE_CONFIG}) ...')
freud.generate_clouds(hub.SG_DIR, pattern=SG_PATTERN, channels=CHANNELS,
                      time_resolutions=TIME_RESOLUTIONS, overwrite=OVERWRITE,
                      extractor_dict=extractor_dict, sg_file_list=sg_file_list,
                      parallel_channel=hub.IN_LINUX)



# Add path in order to be compatible with Linux
import sys, os

SOLUTION_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
print(f'Solution dir = {SOLUTION_DIR}')

PATH_LIST = ['26-HSP', '26-HSP/01-sg-conversion', 'xai-kit', 'hypnomics',
             'xai-kit/roma', 'xai-kit/pictor', 'xai-kit/tframe']

sys.path.append(SOLUTION_DIR)
for p in PATH_LIST: sys.path.append(os.path.join(SOLUTION_DIR, p))

# Import anything here
from freud.hypno_tools.probe_tools import get_probe_keys, get_extractor_dict
from hypnomics.freud.freud import Freud
from roma.console.console import console



# -----------------------------------------------------------------------------
# (1) Configuration
# -----------------------------------------------------------------------------
OVERWRITE = 0
IN_LINUX = os.name != 'nt'

SG_PATH = os.path.join(SOLUTION_DIR, 'data/hsp/hsp_sg')
SG_PATTERN = r'sub-S*_ses-?(float16,128Hz).sg'
NEB_PATH = os.path.join(SOLUTION_DIR, 'data/hsp/hsp_nebula')

CHANNELS = ['EEG F3-M2', 'EEG C3-M2', 'EEG O1-M2',
            'EEG F4-M1', 'EEG C4-M1', 'EEG O2-M1']
TIME_RESOLUTIONS = [30]

PROBE_CONFIG = 'C'

EXTRACTOR_KEYS = get_probe_keys(PROBE_CONFIG)
EXTRACTOR_KEYS.remove('ENTROPY')
extractor_dict = get_extractor_dict(EXTRACTOR_KEYS, fs=128)
# -----------------------------------------------------------------------------
# (2) Cloud generation
# -----------------------------------------------------------------------------
freud = Freud(NEB_PATH)

fs = freud.get_sampling_frequency(SG_PATH, SG_PATTERN, CHANNELS)
assert fs == 128

console.show_status(f'Converting (PROBE_CONFIG={PROBE_CONFIG}) ...')
freud.generate_clouds(SG_PATH, pattern=SG_PATTERN, channels=CHANNELS,
                      time_resolutions=TIME_RESOLUTIONS, overwrite=OVERWRITE,
                      extractor_dict=extractor_dict, parallel_channel=IN_LINUX)


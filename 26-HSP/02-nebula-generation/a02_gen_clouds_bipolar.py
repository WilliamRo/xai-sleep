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
OVERWRITE = 1
IN_LINUX = os.name != 'nt'

if IN_LINUX:
  SG_PATH = os.path.join(SOLUTION_DIR, 'data/hsp/hsp_sg_bipolar')
else:
  SG_PATH = r'\\192.168.5.100\xai-beta\xai-sleep\data\hsp\hsp_sg'

SG_PATTERN = r'sub-S*_ses-?(float16,128Hz,bipolar).sg'

# sub-S0001111531526_ses-4
# SG_PATTERN = r'sub-S*531526_ses-4(float16,128Hz).sg'  # TODO

NEB_PATH = os.path.join(SOLUTION_DIR, 'data/hsp/hsp_nebula')

CHANNELS = ['EEG Fpz-Cz', 'EEG Pz-Oz']
TIME_RESOLUTIONS = [30]

PROBE_CONFIG = 'Ab'
if len(sys.argv) > 1:
  assert len(sys.argv) == 2
  PROBE_CONFIG = sys.argv[1]
assert PROBE_CONFIG in ('A', 'B', 'C', 'AB', 'ABC', 'Ab')

EXTRACTOR_KEYS = get_probe_keys(PROBE_CONFIG)
extractor_dict = get_extractor_dict(EXTRACTOR_KEYS, fs=128)

console.show_status(f'extractor_keys = {list(extractor_dict.keys())}')
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



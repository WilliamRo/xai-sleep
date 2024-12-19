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
from roma import finder



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

PROBE_CONFIG = 'A'
if len(sys.argv) > 1:
  assert len(sys.argv) == 2
  PROBE_CONFIG = sys.argv[1]
assert PROBE_CONFIG in ('A', 'B', 'C', 'AB', 'ABC')

EXTRACTOR_KEYS = get_probe_keys(PROBE_CONFIG)
extractor_dict = get_extractor_dict(EXTRACTOR_KEYS, fs=128)
# -----------------------------------------------------------------------------
# (2) Cloud generation
# -----------------------------------------------------------------------------
freud = Freud(NEB_PATH)

fs = freud.get_sampling_frequency(SG_PATH, SG_PATTERN, CHANNELS)
assert fs == 128

sg_file_list = finder.walk(SG_PATH, pattern=SG_PATTERN)
N = len(sg_file_list)

done_list = []
for i, p in enumerate(sg_file_list):
  console.print_progress(i, N)
  if freud._check_cloud(p, None, CHANNELS, TIME_RESOLUTIONS,
                        extractor_dict):
    done_list.append(p)

n = len(done_list)
console.show_status(f'Progress for probe_config `{PROBE_CONFIG}`: {n}/{N}.')



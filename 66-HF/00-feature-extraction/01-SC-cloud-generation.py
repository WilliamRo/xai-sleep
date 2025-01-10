# Add path in order to be compatible with Linux
import sys, os

SOLUTION_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

PATH_LIST = ['26-HSP', '26-HSP/99-data-probe', 'xai-kit', 'hypnomics', '66-HF',
             'xai-kit/roma', 'xai-kit/pictor', 'xai-kit/tframe']

if __name__ == '__main__':
  print(f'Solution dir = {SOLUTION_DIR}')
  sys.path.append(SOLUTION_DIR)
  for p in PATH_LIST: sys.path.append(os.path.join(SOLUTION_DIR, p))
# -----------------------------------------------------------------------------
from hypnomics.freud.freud import Freud
from hf.extractors import get_extractor_dict
from hf.probe_tools import get_probe_keys



# -----------------------------------------------------------------------------
# (1) Configuration
# -----------------------------------------------------------------------------
CLOUD_DIR = os.path.join(SOLUTION_DIR, 'data/sleepedfx-sc/sc_clouds')
SG_DIR = os.path.join(SOLUTION_DIR, 'data/sleepedfx-sc/sc_sg')

SG_PATTERN = f'*(trim1800;128).sg'

CHANNELS = ['EEG Fpz-Cz', 'EEG Pz-Oz']
TIME_RESOLUTIONS = [
  2,
  5,
  10,
  # 30,
]

PROBE_CONFIG = 'Abc'
EXTRACTOR_KEYS = get_probe_keys(PROBE_CONFIG)

OVERWRITE = 0
# -----------------------------------------------------------------------------
# (2) Cloud generation
# -----------------------------------------------------------------------------
freud = Freud(CLOUD_DIR)

fs = freud.get_sampling_frequency(SG_DIR, SG_PATTERN, CHANNELS)
assert fs == 128

freud.generate_clouds(SG_DIR, pattern=SG_PATTERN, channels=CHANNELS,
                      time_resolutions=TIME_RESOLUTIONS, overwrite=OVERWRITE,
                      extractor_dict=get_extractor_dict(EXTRACTOR_KEYS, fs=fs))

from hypnomics.hypnoprints.probes.wavestats.sun17 import STAT_DICT
from hypnomics.freud.freud import Freud
from hf.extractors import get_extractor_dict



# -----------------------------------------------------------------------------
# (1) Configuration
# -----------------------------------------------------------------------------
WORK_DIR = r'../data/sleepedfx_sc'

SG_DIR = r'../../data/sleep-edf-database-expanded-1.0.0/sleep-cassette'
SG_PATTERN = f'*(trim1800;128).sg'

CHANNELS = [
  'EEG Fpz-Cz',
  'EEG Pz-Oz'
]
TIME_RESOLUTIONS = [
  30,
]
EXTRACTOR_KEYS = [
  'KURT',
]

OVERWRITE = 1
# -----------------------------------------------------------------------------
# (2) Cloud generation
# -----------------------------------------------------------------------------
freud = Freud(WORK_DIR)

fs = freud.get_sampling_frequency(SG_DIR, SG_PATTERN, CHANNELS)
assert fs == 128

freud.generate_clouds(SG_DIR, pattern=SG_PATTERN, channels=CHANNELS,
                      time_resolutions=TIME_RESOLUTIONS, overwrite=OVERWRITE,
                      extractor_dict=get_extractor_dict(EXTRACTOR_KEYS, fs=fs))

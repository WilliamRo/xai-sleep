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
  # 2,
  # 5,
  # 10,
  30,
]
EXTRACTOR_KEYS = [
  'MAG',
  'KURT',
  'ENTROPY',
]

for b1, b2 in [('DELTA', 'TOTAL'), ('THETA', 'TOTAL'), ('ALPHA', 'TOTAL'),
               ('DELTA', 'THETA'), ('DELTA', 'ALPHA'), ('THETA', 'ALPHA')]:
  for stat_key in ['95', 'MIN', 'AVG', 'STD']:
    EXTRACTOR_KEYS.append(f'RPS-{b1}_{b2}_{stat_key}')

for b in ['DELTA', 'THETA', 'ALPHA', 'SIGMA']:
  EXTRACTOR_KEYS.append(f'KURT-{b}')

OVERWRITE = 0
# -----------------------------------------------------------------------------
# (2) Cloud generation
# -----------------------------------------------------------------------------
freud = Freud(WORK_DIR)

fs = freud.get_sampling_frequency(SG_DIR, SG_PATTERN, CHANNELS)
assert fs == 128

freud.generate_clouds(SG_DIR, pattern=SG_PATTERN, channels=CHANNELS,
                      time_resolutions=TIME_RESOLUTIONS, overwrite=OVERWRITE,
                      extractor_dict=get_extractor_dict(EXTRACTOR_KEYS, fs=fs))

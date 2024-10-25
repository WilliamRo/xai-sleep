from hypnomics.freud.freud import Freud
from hf.extractors import get_extractor_dict



# -----------------------------------------------------------------------------
# (1) Configuration
# -----------------------------------------------------------------------------
WORK_DIR = r'../data'

SG_DIR = r'../../data/rrsh-osa'
SG_PATTERN = f'*(trim;simple;100).sg'
# SG_PATTERN = f'318(trim;simple;100).sg'
# TODO: 111 should be excluded

OVERWRITE = 0

CHANNELS = [
  'F3-M2',
  'C3-M2',
  'O1-M2',
  'F4-M1',
  'C4-M1',
  'O2-M1',
]
TIME_RESOLUTIONS = [
  # 2,
  # 5,
  # 10,
  30,
]
EXTRACTOR_KEYS = [
  'AMP-1',
  'FREQ-20',
  'GFREQ-35',
  'P-TOTAL',
  'RP-DELTA',
  'RP-THETA',
  'RP-ALPHA',
  'RP-BETA',
  'MAG',
  'KURT',
  'ENTROPY',
]

for b1, b2 in [('DELTA', 'TOTAL'), ('THETA', 'TOTAL'), ('ALPHA', 'TOTAL'),
               ('DELTA', 'THETA'), ('DELTA', 'ALPHA'), ('THETA', 'ALPHA')]:
  for stat_key in [
    '95',
    'MIN',
    'AVG',
    'STD',
  ]:
    EXTRACTOR_KEYS.append(f'RPS-{b1}_{b2}_{stat_key}')

for b in ['DELTA', 'THETA', 'ALPHA', 'SIGMA']: EXTRACTOR_KEYS.append(f'BKURT-{b}')

OVERWRITE = 0
# -----------------------------------------------------------------------------
# (2) Cloud generation
# -----------------------------------------------------------------------------
freud = Freud(WORK_DIR)

fs = freud.get_sampling_frequency(SG_DIR, SG_PATTERN, CHANNELS)
assert fs == 100

freud.generate_clouds(SG_DIR, pattern=SG_PATTERN, channels=CHANNELS,
                      time_resolutions=TIME_RESOLUTIONS, overwrite=OVERWRITE,
                      extractor_dict=get_extractor_dict(EXTRACTOR_KEYS, fs=fs))

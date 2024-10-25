import os.path

from osaxu.osa_tools import load_nebula_from_clouds
from hypnomics.hypnoprints.extractor import Extractor
from pictor.xomics.omix import Omix
from roma import finder, console

import numpy as np



# -----------------------------------------------------------------------------
# (1) Configuration
# -----------------------------------------------------------------------------
# (1.1) Nebula configuration
WORK_DIR = r'../data'  # contains cloud files
SG_LABELS = finder.walk(WORK_DIR, type_filter='dir', return_basename=True,
                        pattern='*')

# [ 2(x), 5(x), 10(x), 30 ]
TIME_RESOLUTION = 30

CHANNELS = [
  'F3-M2', 'C3-M2', 'O1-M2',
  'F4-M1', 'C4-M1', 'O2-M1',
]

PROBE_KEYS = [
  'FREQ-20',   # 0
  'GFREQ-35',  # 1
  'AMP-1',     # 2
  'P-TOTAL',   # 3
  'RP-DELTA',  # 4
  'RP-THETA',  # 5
  'RP-ALPHA',  # 6
  'RP-BETA',   # 7
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
    PROBE_KEYS.append(f'RPS-{b1}_{b2}_{stat_key}')

for b in ['DELTA', 'THETA', 'ALPHA', 'SIGMA']: PROBE_KEYS.append(f'BKURT-{b}')

# (1.2) Excel path
XLSX_PATH = r"P:\xai-sleep\data\rrsh-osa\OSA-xu.xlsx"

# (1.3) Nebula path
NEB_FN = f'{len(SG_LABELS)}samples-{len(CHANNELS)}channels-{len(PROBE_KEYS)}probes-{TIME_RESOLUTION}s.nebula'
NEB_PATH = os.path.join(WORK_DIR, NEB_FN)

OVERWRITE = 0
# -----------------------------------------------------------------------------
# (2) Load nebula, generate evolution
# -----------------------------------------------------------------------------
if os.path.exists(NEB_PATH) and not OVERWRITE:
  console.show_status(f'Nebula {NEB_PATH} already exists.')
  exit()

nebula = load_nebula_from_clouds(WORK_DIR, SG_LABELS, CHANNELS, TIME_RESOLUTION,
                                 PROBE_KEYS, XLSX_PATH)
nebula.save(NEB_PATH)



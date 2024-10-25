import os

from hypnomics.freud.freud import Freud
from hypnomics.freud.nebula import Nebula
from hypnomics.freud.telescopes.telescope import Telescope
from roma import finder



# -----------------------------------------------------------------------------
# (1) Configuration
# -----------------------------------------------------------------------------
# sca = SCAgent()
# sca.report_data_info()
configs = {
  # 'xmin': 3, 'xmax': 9, 'ymin': 0, 'ymax': 200,
  'show_kde': 0,
  'show_scatter': 0,
  'show_vector': 1,
  # 'scatter_alpha': 0.05,
}

WORK_DIR = r'../data'
CHANNELS = [
  'F3-M2',
  'C3-M2',
  'O1-M2',
  # 'F4-M1',
  # 'C4-M2',
  # 'O2-M1',
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
PK1 = PROBE_KEYS[0]
PK2 = PROBE_KEYS[1]

# Add sun2019 probes
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

# SG_LABELS = ['SC4001E', 'SC4002E']
N = 999
SG_LABELS = finder.walk(WORK_DIR, type_filter='dir', return_basename=True)[:N]

# [ 2, 5, 10, 30, ]
TIME_RESOLUTION = 30

NEB_FN = [
  'None',
  f'OSA-125-30s.nebula',
][1]
# -----------------------------------------------------------------------------
# (2) Visualize
# -----------------------------------------------------------------------------
neb_file_path = os.path.join(WORK_DIR, NEB_FN)
if NEB_FN != 'None' and os.path.exists(neb_file_path):
  nebula: Nebula = Nebula.load(neb_file_path)
else:
  freud = Freud(WORK_DIR)
  nebula = freud.load_nebula(sg_labels=SG_LABELS,
                             channels=CHANNELS,
                             time_resolution=TIME_RESOLUTION,
                             probe_keys=PROBE_KEYS)
  nebula.save(neb_file_path)

viewer_class = Telescope
nebula.dual_view(x_key=PK1, y_key=PK2, viewer_class=viewer_class,
                 viewer_configs={'plotters': 'HA'}, **configs)

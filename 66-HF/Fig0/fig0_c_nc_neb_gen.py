"""Modified from 03-sleep-age/c01_c_nc_omix_gen.py"""
from hf.probe_tools import get_probe_keys
from hf.sc_tools import load_nebula_from_clouds
from roma import finder

import os



# -----------------------------------------------------------------------------
# (1) Nebula Configuration
# -----------------------------------------------------------------------------
# (1.0) Working directory
WORK_DIR = r'../data/sleepedfx_sc'

# (1.1) Patient IDs (.sg labels)
SG_LABELS = finder.walk(WORK_DIR, type_filter='dir', return_basename=True)

# (1.2) Time resolution \in [ 2(x), 5(x), 10(x), 30, ]
TIME_RESOLUTION = 30

# (1.3) Channels
CHANNELS = [
  'EEG Fpz-Cz',
  'EEG Pz-Oz'
]

# (1.4) Probes
PROBE_CONFIG = 'ABC'

# (1.5) Excel path
XLSX_PATH = r'../../data/sleep-edf-database-expanded-1.0.0/SC-subjects.xls'
# -----------------------------------------------------------------------------
# (2) Load nebula, generate evolution
# -----------------------------------------------------------------------------
PROBE_KEYS = get_probe_keys(PROBE_CONFIG)
nebula = load_nebula_from_clouds(WORK_DIR, SG_LABELS, CHANNELS, TIME_RESOLUTION,
                                 PROBE_KEYS, XLSX_PATH)

n_probes = len(nebula.probe_keys)
neb_file_name = f'SC-{TIME_RESOLUTION}s-{PROBE_CONFIG}{n_probes}.nebula'

nebula.save(os.path.join(WORK_DIR, neb_file_name))

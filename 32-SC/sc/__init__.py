# Add path in order to be compatible with Linux
import sys, os

SOLUTION_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

PATH_LIST = ['32-SC', 'xai-kit', 'hypnomics', '66-HF',
             'xai-kit/roma', 'xai-kit/pictor', 'xai-kit/tframe']

print(f'[SC] Solution dir = {SOLUTION_DIR}')
sys.path.append(SOLUTION_DIR)
for p in PATH_LIST: sys.path.append(os.path.join(SOLUTION_DIR, p))

import freud.hypno_tools.probe_tools as probe_tools
import hf.sc_tools as sc_tools

# -----------------------------------------------------------------------------
# (1) Configuration
# -----------------------------------------------------------------------------
IN_LINUX = os.name != 'nt'

DATA_DIR = os.path.join(SOLUTION_DIR, 'data/sleep-edf-database-expanded-1.0.0/sleep-cassette')
SG_DIR = os.path.join(SOLUTION_DIR, 'data/sleepedfx-sc/sc_sg')
CLOUD_DIR = os.path.join(SOLUTION_DIR, 'data/sleepedfx-sc/sc_clouds')
NEBULA_DIR = os.path.join(SOLUTION_DIR, 'data/sleepedfx-sc/sc_nebula')
OMIX_DIR = os.path.join(SOLUTION_DIR, 'data/sleepedfx-sc/sc_omix')
DIST_DIR = os.path.join(SOLUTION_DIR, 'data/sleepedfx-sc/sc_dist')

XLSX_PATH = os.path.join(SOLUTION_DIR, 'data/sleep-edf-database-expanded-1.0.0/SC-subjects.xls')



def get_neb_file_name(time_resolution, probe_config):
  return f'SC-{time_resolution}s-{probe_tools.get_probe_suffix(probe_config)}.nebula'

"""
Last modified: 2024-12-26

This script is for generating nebula for all subsets (bipolar)
"""
# Add path in order to be compatible with Linux
import sys, os

SOLUTION_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

PATH_LIST = ['26-HSP', '26-HSP/99-data-probe', 'xai-kit', 'hypnomics',
             'xai-kit/roma', 'xai-kit/pictor', 'xai-kit/tframe']

if __name__ == '__main__':
  print(f'Solution dir = {SOLUTION_DIR}')
  sys.path.append(SOLUTION_DIR)
  for p in PATH_LIST: sys.path.append(os.path.join(SOLUTION_DIR, p))

from freud.hypno_tools.probe_tools import get_probe_keys
from freud.talos_utils.sleep_sets.hsp import HSPSet
from hypnomics.freud.nebula import Nebula
from roma import io, console

import a00_common as hub



# -----------------------------------------------------------------------------
# (0) Configuration
# -----------------------------------------------------------------------------
OVERWRITE = 0

SUBSET_DICT_FN = hub.SubsetDicts.ss_2ses_3types_bipolar_218

TIME_RESOLUTION = 30
CHANNELS = ['EEG Fpz-Cz', 'EEG Pz-Oz']
PROBE_CONFIG = 'Ab'
PROBE_KEYS = get_probe_keys(PROBE_CONFIG, expand_group=True)

# GAMBLE: n_subjects will be different for all subsets
n_subjects = int(SUBSET_DICT_FN.split('_')[-1].split('.')[0])
m_subjects = n_subjects
NEB_FN = f'HSP-{m_subjects}-{PROBE_CONFIG}-{len(CHANNELS)}chn-{TIME_RESOLUTION}s.nebula'
NEB_PATH = os.path.join(hub.OMIX_DIR, NEB_FN)
# -----------------------------------------------------------------------------
# (1) Load subset
# -----------------------------------------------------------------------------
subset_dict = hub.ha.load_subset_dict(SUBSET_DICT_FN, max_subjects=m_subjects)

# -----------------------------------------------------------------------------
# (2) Load nebula, add acq_time and age as properties
# -----------------------------------------------------------------------------
if not OVERWRITE and os.path.exists(NEB_PATH):
  nebula: Nebula = Nebula.load(NEB_PATH, verbose=True)
else:
  nebula = hub.ha.load_nebula_from_clouds(subset_dict, hub.CLOUD_DIR, CHANNELS,
                                          TIME_RESOLUTION, PROBE_KEYS)

  nebula.save(NEB_PATH, verbose=True)

# -----------------------------------------------------------------------------
# (3) Visualize nebula
# -----------------------------------------------------------------------------
if not hub.IN_LINUX:
  from hypnomics.freud.telescopes.telescope import Telescope
  from hypnomics.freud.telescopes.popglass import PopGlass

  # PK1, PK2 = 'FREQ-20', 'AMP-1'
  # PK1, PK2 = 'FREQ-20', 'PR-THETA_TOTAL'
  # PK1, PK2 = 'FREQ-20', 'PR-ALPHA_TOTAL'
  PK1, PK2 = 'PR-ALPHA_TOTAL', 'PR-DELTA_TOTAL'

  viewer_class = [Telescope, PopGlass][1]

  if viewer_class is Telescope:
    configs = {
      'show_kde': 0,
      'show_scatter': 1,
      'show_vector': 0,
    }
    nebula.dual_view(x_key=PK1, y_key=PK2, viewer_class=viewer_class,
                     viewer_configs={'plotters': 'HA'}, **configs)
  else:
    nebula.dual_view(x_key=PK1, y_key=PK2, viewer_class=viewer_class)

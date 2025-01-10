# Add path in order to be compatible with Linux
import sys, os

import numpy as np

SOLUTION_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
print(f'Solution dir = {SOLUTION_DIR}')

PATH_LIST = ['26-HSP', '26-HSP/01-sg-conversion', 'xai-kit',
             'xai-kit/roma', 'xai-kit/pictor', 'xai-kit/tframe']

sys.path.append(SOLUTION_DIR)
for p in PATH_LIST: sys.path.append(os.path.join(SOLUTION_DIR, p))

# Import anything here
from freud.gui.freud_gui import Freud
from roma import console
from roma import io, finder



# -----------------------------------------------------------------------------
# (1) Configuration
# -----------------------------------------------------------------------------
SG_DIR = os.path.join(SOLUTION_DIR, 'data/hsp/hsp_sg_bipolar')
SG_PATTERN = r'sub-S*_ses-?(float16,128Hz,bipolar).sg'


# Number of .sg files to visualize
N = 10

# -----------------------------------------------------------------------------
# (2) Get folder list
# -----------------------------------------------------------------------------
sg_path_list = finder.walk(SG_DIR, pattern=SG_PATTERN)[:N]

console.show_status('Loading ...')
signal_groups = [io.load_file(p, verbose=True) for p in sg_path_list]

console.show_status('Visualizing ...')
Freud.visualize_signal_groups(
  signal_groups, title='HSP', default_win_duration=9999999,
)

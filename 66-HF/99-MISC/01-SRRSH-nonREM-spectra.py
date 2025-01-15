# Add path in order to be compatible with Linux
import sys, os

SOLUTION_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

PATH_LIST = ['31-OSA-XU', 'dev/tools',
             'xai-kit/roma', 'xai-kit/pictor', 'xai-kit/tframe']

if __name__ == '__main__':
  print(f'Solution dir = {SOLUTION_DIR}')
  sys.path.append(SOLUTION_DIR)
  for p in PATH_LIST: sys.path.append(os.path.join(SOLUTION_DIR, p))

# -----------------------------------------------------------------------------
from spectra_explorer import SpectraExplorer, SignalGroup
from roma import io, finder

import numpy as np
import osaxu as hub



# -----------------------------------------------------------------------------
# (1) Configuration
# -----------------------------------------------------------------------------
SG_PATTERN = r'*(trim;simple;100).sg'
CHANNELS = ['F3-M2', 'C3-M2', 'O1-M2', 'F4-M1', 'C4-M1', 'O2-M1']

N = 999


# -----------------------------------------------------------------------------
# (2) Get sg list
# -----------------------------------------------------------------------------
sg_file_names = finder.walk(hub.SG_DIR, 'file', SG_PATTERN)[:N]

sg_list = [io.load_file(p, verbose=True) for p in sg_file_names]
sg_labels = [sg.label for sg in sg_list]

_meta = hub.osa_tools.load_osa_meta(hub.XLSX_PATH, sg_labels)
meta = {}
for lb in sg_labels:
  info = f'{_meta[lb]["age"]:.0f}y, {"M" if _meta[lb]["gender"] else "F"}'
  info += f', AHI={_meta[lb]["AHI"]:.0f}'

  for k1, k2 in [('cog_imp', 'CI'), ('dep', 'D'), ('anx', 'A'), ('som', 'S')]:
    v = _meta[lb][k1]
    if np.isnan(v): continue
    if v == 0: info += f', N{k2}'
    else: info += f', {k2}'

  meta[lb] = {'info': info}

# -----------------------------------------------------------------------------
# (3) Visualize
# -----------------------------------------------------------------------------
# Visualize signal groups
ee = SpectraExplorer.explore(sg_list, channels=CHANNELS, meta=meta,
                             figure_size=(7, 5))


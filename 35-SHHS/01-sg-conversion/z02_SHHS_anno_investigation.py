# Add path in order to be compatible with Linux
import sys, os

SOLUTION_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

PATH_LIST = ['35-SHHS', 'xai-kit', 'hypnomics', '66-HF',
             'xai-kit/roma', 'xai-kit/pictor', 'xai-kit/tframe']

print(f'[SHHS] Solution dir = {SOLUTION_DIR}')
sys.path.append(SOLUTION_DIR)
for p in PATH_LIST: sys.path.append(os.path.join(SOLUTION_DIR, p))

# -----------------------------------------------------------------------------
from roma import console

import shhs as hub
import mne.io



# -----------------------------------------------------------------------------
# (1) Configuration
# -----------------------------------------------------------------------------
N = len(hub.sa.actual_two_visits_dict)

# -----------------------------------------------------------------------------
# (2) Do statistics
# -----------------------------------------------------------------------------
channel_numbers = {}

for i, pid in enumerate(hub.sa.actual_two_visits_dict.keys()):
  if i >= N: break
  console.print_progress(i, N)

  for sid in ('1', '2'):
    edf_path, _ = hub.sa.get_edf_anno_by_id(pid, sid)

    if not os.path.exists(edf_path):
      console.warning(f'`{edf_path}` not exist.')
      continue

    with mne.io.read_raw_edf(edf_path, preload=False, verbose=False) as file:
      for ck in file.ch_names:
        if ck not in channel_numbers: channel_numbers[ck] = 0
        channel_numbers[ck] += 1

console.show_info(f'Investigated {N} patients:')
sorted_keys = sorted(channel_numbers.keys(), key=lambda k: channel_numbers[k],
                     reverse=True)
for key in sorted_keys: console.supplement(f'{key}: {channel_numbers[key]}')

console.show_status(f'EEG # = {channel_numbers["EEG"]}')
keys = ('EEG(SEC)', 'EEG sec', 'EEG2', 'EEG 2', 'EEG(sec)')
m = sum([channel_numbers[k] for k in keys])
console.show_status(f'sum({keys}) # = {m}')

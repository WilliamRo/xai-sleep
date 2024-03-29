from freud.talos_utils.slp_set import SleepSet
from roma import console

import os
import time



data_root = r'../../../../../data/sleepedfx'
edf_path = r'SC4001E0-PSG.edf'
file_path = os.path.join(data_root, edf_path)

# -----------------------------------------------------------------------------
# Grouped
# -----------------------------------------------------------------------------
tic = time.time()
groups = [['EEG Fpz-Cz', 'EEG Pz-Oz', 'EOG horizontal'],
          ['Resp oro-nasal', 'EMG submental', 'Temp rectal', 'Event marker']]
digital_signals = SleepSet.read_digital_signals_mne(file_path, groups)
for ds in digital_signals: print(ds)
console.show_info(f'Grouped time elapsed: {time.time()-tic:.2}s')

console.split()

# -----------------------------------------------------------------------------
# Ungrouped
# -----------------------------------------------------------------------------
tic = time.time()
digital_signals = SleepSet.read_digital_signals_mne(file_path)
for ds in digital_signals: print(ds)
console.show_info(f'Ungrouped time elapsed: {time.time()-tic:.2}s')




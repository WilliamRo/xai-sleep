from freud.talos_utils.slp_set import SleepSet
from roma import console

import os
import time



data_root = r'../../../../../data/ucddb'
edf_path = r'ucddb002.rec.edf'
file_path = os.path.join(data_root, edf_path)


tic = time.time()
digital_signals = SleepSet.read_digital_signals_mne(
  file_path, allow_rename=True)
for ds in digital_signals: print(ds)
console.show_info(f'Time elapsed: {time.time()-tic:.2}s')


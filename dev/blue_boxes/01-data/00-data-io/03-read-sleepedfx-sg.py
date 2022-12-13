from freud.talos_utils.sleep_sets.sleepedfx import SleepEDFx
from roma import console

import time



data_root = r'../../../../data/sleepedf'

tic = time.time()
signal_groups = SleepEDFx.load_as_signal_groups(
  data_root,
  save_sg=True,
  overwrite=0
)

time_elapsed = time.time() - tic
console.show_status(
  f'Reading {len(signal_groups)} sg files using {time_elapsed:.2f} seconds')

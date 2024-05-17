from freud.talos_utils.sleep_sets.sleepedfx import SleepEDFx
from tframe import console

import time



console.suppress_logging()
data_dir = r'../../../data/sleep-edf-database-expanded-1.0.0/sleep-telemetry'

tic = time.time()
preprocess = 'trim,1800;128'

fn_pattern = '*ST*'
SleepEDFx.load_as_sleep_set(
  data_dir, overwrite=0, fn_pattern=fn_pattern, preprocess=preprocess,
  overwrite_pp=0, just_conversion=True)

elapsed = time.time() - tic
console.show_info(f'Time elapsed = {elapsed:.2f} sec.')

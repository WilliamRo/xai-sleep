"""

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

from roma import io

import a00_common as hub



# -----------------------------------------------------------------------------
# (0) Configuration
# -----------------------------------------------------------------------------
# TODO: configure here
subset_dict_fn = hub.SubsetDicts.ss_2ses_3types_378
subset_dict_path = os.path.join(SOLUTION_DIR, 'data/hsp', subset_dict_fn)

# -----------------------------------------------------------------------------
# (1) Load subset dict
# -----------------------------------------------------------------------------
assert os.path.exists(subset_dict_path)
patient_dict = io.load_file(subset_dict_path, verbose=True)

N = sum([len(v) for v in patient_dict.values()])
hub.console.show_status(
  f'There are {len(patient_dict)} patients with at least 2 sessions with '
  f'annotation, altogether {N} folders.')
# -----------------------------------------------------------------------------
# (2) Attach acquisition time
# -----------------------------------------------------------------------------
if not hub.ha.check_acq_time_in_pd(patient_dict):
  for pid, ses_dict in patient_dict.items():
    for sid, sg_dict in ses_dict.items():
      ses_path = hub.ha.get_ses_path(pid, sid)
      sg_dict['acq_time'] = hub.ha.get_acq_time(ses_path, return_str=False)

  # Save subset dict
  io.save_file(patient_dict, subset_dict_path, verbose=True)

assert hub.ha.check_acq_time_in_pd(patient_dict)
hub.console.show_status(f'Acquisition time included in `{subset_dict_fn}`.')










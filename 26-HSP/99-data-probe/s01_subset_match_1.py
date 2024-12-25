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

from a00_common import ha, console, SubsetDicts, SG_DIR, CLOUD_DIR
from roma import io

import numpy as np



# -----------------------------------------------------------------------------
# (0) Configuration
# -----------------------------------------------------------------------------
subset_dict_fn = SubsetDicts.ss_2ses_3types_378
subset_dict_path = os.path.join(SOLUTION_DIR, 'data/hsp', subset_dict_fn)

OVERWRITE = 0

if os.path.exists(subset_dict_path) and not OVERWRITE:
  patient_dict = io.load_file(subset_dict_path, verbose=True)
else:
  # -----------------------------------------------------------------------------
  # (1) Filter by META
  # -----------------------------------------------------------------------------
  study_types = ['PSG Diagnostic', 'Master', 'PSG']
  patient_dict = ha.filter_patients_meta(
    min_n_sessions=2, should_have_annotation=True, study_types=study_types)
  N1 = sum([len(v) for v in patient_dict.values()])

  console.show_status(
    f'There are {len(patient_dict)} patients with at least 2 sessions with '
    f'annotation, altogether {N1} folders.', prompt='[META]')

  # -----------------------------------------------------------------------------
  # (2) Filter by .sg files
  # -----------------------------------------------------------------------------
  patient_dict = ha.filter_patients_sg(
    patient_dict, SG_DIR, min_n_sessions=2, dtype=np.float16, max_sfreq=128,
    verbose=True)

  N2 = sum([len(v) for v in patient_dict.values()])
  console.show_status(
    f'There are {len(patient_dict)} patients with at least 2 sessions with '
    f'annotation, altogether {N2} folders.', prompt='[SG]')

  # -----------------------------------------------------------------------------
  # (3) Filter by .clouds files
  # -----------------------------------------------------------------------------
  patient_dict = ha.filter_patients_neb(
    patient_dict, CLOUD_DIR, min_n_sessions=2, verbose=True, min_hours=2)

  io.save_file(patient_dict, subset_dict_path, verbose=True)

# -----------------------------------------------------------------------------
# (*) Report
# -----------------------------------------------------------------------------
N3 = sum([len(v) for v in patient_dict.values()])
console.show_status(
  f'There are {len(patient_dict)} patients with at least 2 sessions with '
  f'annotation, altogether {N3} folders.', prompt='[NEB]')

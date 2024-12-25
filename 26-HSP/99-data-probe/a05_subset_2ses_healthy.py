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

from a00_common import ha, console



# -----------------------------------------------------------------------------
# (1) Plot histogram of study types
# -----------------------------------------------------------------------------
study_types = [
  'PSG Diagnostic', 'Master', 'PSG',
]
patient_dict = ha.filter_patients_meta(
  min_n_sessions=2, should_have_annotation=True, should_have_psq=0,
  study_types=study_types)

console.show_status(f'{len(patient_dict)} subjects with {study_types}.')


"""
>> At least 2 sessions, with annotation, 
   .. with ['PSG Diagnostic', 'Master', 'PSG'], 631 subjects 
>> At least 2 sessions, with annotation, with PSQ,
   .. with ['PSG Diagnostic', 'Master', 'PSG'], 17 subjects 
>> At least 2 sessions, with annotation, with PSQ: 1927 subjects
"""


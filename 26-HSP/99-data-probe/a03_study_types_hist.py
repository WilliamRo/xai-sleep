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
patient_dict = ha.filter_patients_meta(
  min_n_sessions=2, should_have_annotation=True)

study_type_dict = {}
for pid, ses_dict in patient_dict.items():
  for ses_id, ses_info in ses_dict.items():
    study_type = ses_info['study_type']
    if study_type not in study_type_dict: study_type_dict[study_type] = 0
    study_type_dict[study_type] += 1

# Sort with number
study_type_dict = dict(sorted(
  study_type_dict.items(), key=lambda x: x[1], reverse=True))

console.show_status(f'Total {len(study_type_dict)} study types.')
for k, v in study_type_dict.items():
  console.show_status(f'{k}: {v}')

"""
>> Total 89 study types.
>> PSG all night CPAP: 3287
>> PSG Diagnostic : 2992
>> Master: 1678
>> PSG Split night: 1518
>> PSG: 723
>> MSLT: 345
>> PSG Diagnostic: 323
>> Diagnostic: 81
>> Extended EEG-sleep montage: 44
>> Titration CPAP: 32
>> Split Night: 29
>> CPAP titration: 27
>> CPAP: 17
>> Split night: 14
>> Extend: 12
>> Master - merged: 10
"""

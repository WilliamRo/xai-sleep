"""Check whether all folders listed in META file exist in the local storage.

HSP homepage: The Human Sleep Project dataset includes 26,200 PSG studies
              conducted on 19,492 distinct patients, as outlined below: ...
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
# (1) Check compactness
# -----------------------------------------------------------------------------
VERBOSE = len(sys.argv) > 1 and int(sys.argv[1]) > 0

console.show_status(f'HSP includes {len(ha.patient_dict)} patients in total.',
                    prompt='[META]')

path_list = ha.convert_to_folder_names(ha.patient_dict, local=True)
N = len(path_list)
m = 0

for i, path in enumerate(path_list):
  if i % 100 == 0: console.print_progress(i, N)
  if os.path.exists(path): m += 1

console.show_status(f'{N - m} out of {N} folders not exist.', prompt='[CHECK]')

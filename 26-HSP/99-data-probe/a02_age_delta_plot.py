# Add path in order to be compatible with Linux
import sys, os

SOLUTION_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

PATH_LIST = ['26-HSP', '26-HSP/99-data-probe', 'xai-kit', 'hypnomics',
             'xai-kit/roma', 'xai-kit/pictor', 'xai-kit/tframe']

if __name__ == '__main__':
  print(f'Solution dir = {SOLUTION_DIR}')
  sys.path.append(SOLUTION_DIR)
  for p in PATH_LIST: sys.path.append(os.path.join(SOLUTION_DIR, p))

from a00_common import ha, console, DATA_DIR
from freud.talos_utils.sleep_sets.hsp import HSPOrganization

from datetime import datetime
import pandas as pd



# -----------------------------------------------------------------------------
# (1) Get data for plot
# -----------------------------------------------------------------------------
patient_dict = ha.filter_patients_meta(
  min_n_sessions=2, should_have_annotation=True)

ses_path_act_dict = {}

def get_acq_time(ses_path):
  if ses_path not in ses_path_act_dict:
    ses_path_act_dict[ses_path] = ha.get_acq_time(ses_path)
  return ses_path_act_dict[ses_path]

X, Y = [], []
for cursor, (pid, ses_dict) in enumerate(patient_dict.items()):
  if cursor % 50 == 0: console.print_progress(cursor, len(patient_dict))

  ses_ids = sorted(ses_dict.keys())
  for i in range(len(ses_ids) - 1):
    #
    id_i = ses_ids[i]
    ses_path_i = os.path.join(DATA_DIR, ses_dict[id_i]['bids_folder'],
                              'ses-' + str(id_i))
    act_1 = get_acq_time(ses_path_i)
    if act_1 is None: continue

    for j in range(i + 1, len(ses_ids)):
      x = ses_dict[ses_ids[i]]['age']

      id_j = ses_ids[j]
      ses_path_j = os.path.join(DATA_DIR, ses_dict[id_j]['bids_folder'],
                                'ses-' + str(id_j))
      act_2 = get_acq_time(ses_path_j)
      if act_2 is None: continue
      y = (act_2 - act_1).days / 365.25

      if y < 0:
        console.show_status(f'{pid}({ses_ids[i]}->{ses_ids[j]})', prompt='[NEGATIVE]')
        y = ses_dict[ses_ids[j]]['age'] - x

      X.append(x)
      Y.append(y)

console.show_status('Plotting ...')

# (*) report the number of pairs within min_interval
for mi in (0.1, 0.2, 0.3, 0.4, 0.5):
  n_pairs = sum([1 for y in Y if abs(y) < mi])
  console.show_status(f'{n_pairs} within {mi} years.')

# -----------------------------------------------------------------------------
# (2) Plot
# -----------------------------------------------------------------------------
import matplotlib.pyplot as plt


fig, ax = plt.subplots(1, 1, figsize=(12, 6))
ax.scatter(X, Y, alpha=0.5)
ax.set_xlabel('Age (years)')
ax.set_ylabel('Age difference (years)')
ax.set_title(f'{len(X)} pairs')

ax.grid(True)
plt.tight_layout()
plt.show()



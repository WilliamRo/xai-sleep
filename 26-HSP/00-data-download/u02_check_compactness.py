"""Check whether all folders listed in META file (filtered) exist in the local storage."""
import sys, os
SOLUTION_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

from freud.talos_utils.sleep_sets.hsp import HSPAgent, HSPOrganization
from roma.console.console import console



# -----------------------------------------------------------------------------
# (1) Configuration
# -----------------------------------------------------------------------------
OVERWRITE = 0
IN_LINUX = os.name != 'nt'
assert not IN_LINUX

META_DIR = os.path.join(SOLUTION_DIR, 'data/hsp')
DATA_DIR = r'E:\data\hsp_raw'
META_TIME_STAMP = '20231101'

# -----------------------------------------------------------------------------
# (2) Select path list to check
# -----------------------------------------------------------------------------
ha = HSPAgent(META_DIR, DATA_DIR, META_TIME_STAMP)

pd = ha.filter_patients_meta(min_n_sessions=2, should_have_annotation=True)
path_list = ha.convert_to_folder_names(pd, local=1)

# -----------------------------------------------------------------------------
# (3) Check compactness
# -----------------------------------------------------------------------------
n, N = 0, len(path_list)
n_tsv = 0

for i, p in enumerate(path_list):
  if i % 50 == 0: console.print_progress(i, N)

  ho = HSPOrganization(p)

  if os.path.exists(p): n += 1
  if os.path.exists(ho.tsv_path):
    n_tsv += 1

    # Read tsv file
    # import pandas as pd
    # df = pd.read_csv(ho.tsv_path, sep='\t')
    # date_str = df['acq_time'].str.split('T').str[0].iloc[0]
    # print(date_str)
  else:
    console.warning(f'{ho.tsv_path} not found.')

console.show_status(f'{n}/{N} folders detected.')
console.show_status(f'{n_tsv}/{N} .tsv files detected.')

"""
>> 11261/11261 folders detected.
>> 11158/11261 .tsv files detected.
"""

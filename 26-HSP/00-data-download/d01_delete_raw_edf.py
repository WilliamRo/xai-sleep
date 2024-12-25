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
# (2) Delete all edf file
# -----------------------------------------------------------------------------
ha = HSPAgent(META_DIR, DATA_DIR, META_TIME_STAMP)

path_list = ha.convert_to_folder_names(ha.patient_dict, local=1)
m, N = 0, len(path_list)
for i, p in enumerate(path_list):
  if i % 50 == 0: console.print_progress(i, N)

  edf_path = HSPOrganization(p).edf_path

  if os.path.exists(edf_path):
    # TODO: uncomment this line to delete edf files
    # os.remove(edf_path)
    m += 1

console.show_status(f'{m}/{N} .edf files removed.')

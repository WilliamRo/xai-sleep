# Add path in order to be compatible with Linux
import sys, os

SOLUTION_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
print(f'Solution dir = {SOLUTION_DIR}')

PATH_LIST = ['26-HSP', '26-HSP/01-sg-conversion', 'xai-kit',
             'xai-kit/roma', 'xai-kit/pictor', 'xai-kit/tframe']

sys.path.append(SOLUTION_DIR)
for p in PATH_LIST: sys.path.append(os.path.join(SOLUTION_DIR, p))

# Import anything here
from freud.talos_utils.sleep_sets.hsp import HSPAgent, HSPSet, HSPOrganization
from pictor.objects.signals.signal_group import SignalGroup, Annotation
from roma import console, finder, io



# -----------------------------------------------------------------------------
# (1) Configuration
# -----------------------------------------------------------------------------
TGT_PATH = os.path.join(SOLUTION_DIR, 'data/hsp/hsp_sg')
SG_PATTERN = r'sub-S*_ses-?(float16,128Hz).sg'

assert os.name != 'nt'
# -----------------------------------------------------------------------------
# (2) Traverse folder
# -----------------------------------------------------------------------------
sg_file_list = finder.walk(TGT_PATH, pattern=SG_PATTERN)

N = 0
for i, file_path in enumerate(sg_file_list):
  sg: SignalGroup = io.load_file(file_path, verbose=False)

  base_name = os.path.basename(file_path)
  if sg.label not in base_name:
    console.warning(f'File (`{base_name}`) does not match label (`{sg.label}`).')
    # Delete file
    os.remove(file_path)
    console.show_status(f'File {file_path} has been deleted.')
    N += 1

  console.print_progress(i, len(sg_file_list))

console.show_status(f'Deleted {N} mismatched files.')


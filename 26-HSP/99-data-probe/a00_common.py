# Add path in order to be compatible with Linux
import sys, os

SOLUTION_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
print(f'Solution dir = {SOLUTION_DIR}')

PATH_LIST = ['26-HSP', '26-HSP/01-sg-conversion', 'xai-kit', 'hypnomics',
             'xai-kit/roma', 'xai-kit/pictor', 'xai-kit/tframe']

if __name__ == '__main__':
  sys.path.append(SOLUTION_DIR)
  for p in PATH_LIST: sys.path.append(os.path.join(SOLUTION_DIR, p))

from freud.talos_utils.sleep_sets.hsp import HSPAgent
from roma.console.console import console



# -----------------------------------------------------------------------------
# (1) Configuration
# -----------------------------------------------------------------------------
OVERWRITE = 0
IN_LINUX = os.name != 'nt'

if IN_LINUX:
  console.show_status('Linux system detected.')
  SG_PATH = os.path.join(SOLUTION_DIR, 'data/hsp/hsp_sg')
  DATA_DIR = os.path.join(SOLUTION_DIR, r'\\192.168.5.100\xai-beta\xai-sleep\data\hsp\hsp_raw')
else:
  console.show_status('Windows system detected.')
  SG_PATH = r'\\192.168.5.100\xai-beta\xai-sleep\data\hsp\hsp_sg'
  DATA_DIR = os.path.join(SOLUTION_DIR, 'data/hsp/hsp_raw')

META_DIR = os.path.join(SOLUTION_DIR, 'data/hsp')
META_TIME_STAMP = '20231101'
META_PATH = os.path.join(META_DIR, f'bdsp_psg_master_{META_TIME_STAMP}.csv')


# -----------------------------------------------------------------------------
# (1) Configuration
# -----------------------------------------------------------------------------
ha = HSPAgent(META_DIR, DATA_DIR, META_TIME_STAMP)
console.show_status(f'Total subject number: {len(ha.patient_dict)}',
                    prompt='[META]')

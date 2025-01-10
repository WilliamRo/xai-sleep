# Add path in order to be compatible with Linux
import sys, os

SOLUTION_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

PATH_LIST = ['26-HSP', '26-HSP/99-data-probe', 'xai-kit', 'hypnomics',
             'xai-kit/roma', 'xai-kit/pictor', 'xai-kit/tframe']

if __name__ == '__main__':
  print(f'Solution dir = {SOLUTION_DIR}')
  sys.path.append(SOLUTION_DIR)
  for p in PATH_LIST: sys.path.append(os.path.join(SOLUTION_DIR, p))

from freud.talos_utils.sleep_sets.hsp import HSPAgent
from roma.console.console import console



# -----------------------------------------------------------------------------
# (1) Configuration
# -----------------------------------------------------------------------------
OVERWRITE = 0
IN_LINUX = os.name != 'nt'

# TODO: Set HSP_RAW path
DATA_DIR = os.path.join(SOLUTION_DIR, 'data/hsp/hsp_raw')
# DATA_DIR = r'E:\data\hsp_raw'  # This is for home-study

SG_DIR = os.path.join(SOLUTION_DIR, 'data/hsp/hsp_sg')

META_DIR = os.path.join(SOLUTION_DIR, 'data/hsp')
META_TIME_STAMP = '20231101'
META_PATH = os.path.join(META_DIR, f'bdsp_psg_master_{META_TIME_STAMP}.csv')

CLOUD_DIR = os.path.join(SOLUTION_DIR, 'data/hsp/hsp_nebula')
OMIX_DIR = os.path.join(SOLUTION_DIR, 'data/hsp/hsp_omix')
MATCH_DIR = os.path.join(SOLUTION_DIR, 'data/hsp/hsp_match')
MATCH_PI = os.path.join(SOLUTION_DIR, 'data/hsp/hsp_pi')
DIST_DIR = os.path.join(SOLUTION_DIR, 'data/hsp/hsp_dist')

if IN_LINUX:
  console.show_status('Linux system detected.')
else:
  console.show_status('Windows system detected.')
  CLOUD_DIR = r'F:/data/hsp/hsp_nebula'
  SG_DIR = f'F:/data/hsp/hsp_sg'
  # SG_DIR = r'\\192.168.5.100\xai-beta\xai-sleep\data\hsp\hsp_sg'
  DATA_DIR = f'F:/data/hsp/hsp_raw'

class SubsetDicts:
  ss_2ses_3types_378 = 'subset_2ses_3types_378.od'
  ss_2ses_3types_bipolar_218 = 'subset_2ses_3types_bipolar_218.od'

# -----------------------------------------------------------------------------
# (1) Configuration
# -----------------------------------------------------------------------------
ha = HSPAgent(META_DIR, DATA_DIR, META_TIME_STAMP)



if __name__ == '__main__':
  # ha.put_into_pocket('OVERWRITE_PSQQ', None)
  # ha.put_into_pocket('OVERWRITE_PD', None)

  console.show_status(f'Total subject number: {len(ha.patient_dict)}',
                      prompt='[META]')

  od = ha.pre_sleep_questionnaire_dict
  console.show_status(f'{len(od)} subjects have pre-sleep questionnaire data.')

  df = ha.pre_sleep_questionnaire_dataframe
  psq_csv_path = ha.meta_path.replace('.csv', '_psq.csv')
  df.replace({True: '1', False: '0', 'TRUE': 1, 'FALSE': 0})
  df.to_csv(psq_csv_path, index=True)

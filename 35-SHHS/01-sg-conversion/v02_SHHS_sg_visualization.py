# Add path in order to be compatible with Linux
import sys, os

SOLUTION_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

PATH_LIST = ['35-SHHS', 'xai-kit', 'hypnomics', '66-HF',
             'xai-kit/roma', 'xai-kit/pictor', 'xai-kit/tframe']

print(f'[SHHS] Solution dir = {SOLUTION_DIR}')
sys.path.append(SOLUTION_DIR)
for p in PATH_LIST: sys.path.append(os.path.join(SOLUTION_DIR, p))

# -----------------------------------------------------------------------------
import shhs as hub



# -----------------------------------------------------------------------------
# (1) Configuration
# -----------------------------------------------------------------------------
edf_path, anno_path = hub.sa.get_edf_anno_by_id('200001', '1')
sg: hub.SignalGroup = hub.SHHSet.load_sg_from_raw_files(edf_path, anno_path)


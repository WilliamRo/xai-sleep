from sc.sc_agent import SCAgent
from hf.cloud_viewer import view_fingerprints



# -----------------------------------------------------------------------------
# (1) Configuration
# -----------------------------------------------------------------------------
# sca = SCAgent()
# sca.report_data_info()
configs = {
  # 'xmin': 3, 'xmax': 9, 'ymin': 0, 'ymax': 200,
  'show_kde': False,
  # 'show_vector': True,
  # 'scatter_alpha': 0.05,
}

WORK_DIR = r'../../data/sleepedfx_sc'
CHANNELS = [
  'EEG Fpz-Cz',
  'EEG Pz-Oz'
]
TIME_RESOLUTION = 30
PK1 = 'FREQ-20'
PK2 = 'AMP-1'

SG_LABELS = [
  'SC4001E',
  'SC4002E',
]
# -----------------------------------------------------------------------------
# (2) Visualize
# -----------------------------------------------------------------------------
view_fingerprints(WORK_DIR, SG_LABELS, CHANNELS, TIME_RESOLUTION, PK1, PK2,
                  **configs)

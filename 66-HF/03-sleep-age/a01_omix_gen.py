from hf.sc_tools import load_nebula_from_clouds
from hypnomics.hypnoprints.extractor import Extractor
from pictor.xomics.omix import Omix
from roma import finder

import numpy as np



# -----------------------------------------------------------------------------
# (1) Configuration
# -----------------------------------------------------------------------------
# (1.1) Nebula configuration
WORK_DIR = r'../data/sleepedfx_sc'
SG_LABELS = finder.walk(WORK_DIR, type_filter='dir', return_basename=True)

# [ 2(x), 5(x), 10(x), 30, ]
TIME_RESOLUTION = 30

CHANNELS = [
  'EEG Fpz-Cz',
  'EEG Pz-Oz'
]

PROBE_KEYS = [
  'FREQ-20',   # 0
  # 'GFREQ-35',  # 1
  'AMP-1',     # 2
  'P-TOTAL',   # 3
  'RP-DELTA',  # 4
  'RP-THETA',  # 5
  'RP-ALPHA',  # 6
  'RP-BETA',   # 7
]

# (1.3) Excel path
XLSX_PATH = r'../../data/sleep-edf-database-expanded-1.0.0/SC-subjects.xls'

# (1.4)
E_SETTINGS = {
  # 'include_proportion': 0,
  # 'include_stage_shift': 0,
  # 'include_channel_shift': 0,
  # 'include_stage_wise_covariance': 0,
  # 'include_stage_mean': 1,
}
# -----------------------------------------------------------------------------
# (2) Load nebula, generate evolution
# -----------------------------------------------------------------------------
nebula = load_nebula_from_clouds(WORK_DIR, SG_LABELS, CHANNELS, TIME_RESOLUTION,
                                 PROBE_KEYS, XLSX_PATH)

# -----------------------------------------------------------------------------
# (3) Visualization
# -----------------------------------------------------------------------------
_gender = [None, 'male', 'female'][0]



if __name__ == '__main__':
  extractor = Extractor(**E_SETTINGS)
  feature_dict = extractor.extract(nebula, return_dict=True)
  features = np.stack([np.array(list(v.values()))
                       for v in feature_dict.values()], axis=0)
  feature_names = list(list(feature_dict.values())[0].keys())

  target_labels = ['Age']
  targets = [nebula.meta[pid]['age'] for pid in nebula.labels]
  omix = Omix(features, targets, feature_names, nebula.labels, target_labels,
              data_name=f'SC-age-153x375-{TIME_RESOLUTION}s')

  if _gender is not None:
    indices = [pid for pid in nebula.labels
               if nebula.meta[pid]['gender'] == _gender]
    data_name = f'SC-age-{len(indices)}x375-{TIME_RESOLUTION}s-{_gender}'
    omix = omix.select_samples(indices, data_name)

  omix.show_in_explorer()

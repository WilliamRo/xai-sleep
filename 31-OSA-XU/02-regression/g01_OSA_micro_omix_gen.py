"""See g03_OSA_hypno_omix_gen.py

"""
from hypnomics.freud.nebula import Nebula
from hypnomics.hypnoprints.extractor import Extractor
from osaxu.osa_tools import set_target_collection_for_omix
from pictor.xomics.omix import Omix
from roma import finder

import numpy as np
import os



# -----------------------------------------------------------------------------
# (1) Configuration
# -----------------------------------------------------------------------------
# (1.1) Nebula configuration
WORK_DIR = r'../data'  # contains cloud files
SG_LABELS = finder.walk(WORK_DIR, type_filter='dir', return_basename=True,
                        pattern='*')

# (1.2) Nebula path
NEB_FN = f'125samples-6channels-39probes-30s.nebula'
NEB_PATH = os.path.join(WORK_DIR, NEB_FN)

# (1.3) Set default target
TARGET = 'AHI'

OVERWRITE = 0
# -----------------------------------------------------------------------------
# (2)
# -----------------------------------------------------------------------------
OMIX_PATH = NEB_PATH.replace('.nebula', '.omix')

if os.path.exists(OMIX_PATH) and not OVERWRITE:
  omix = Omix.load(OMIX_PATH)
else:
  nebula = Nebula.load(NEB_PATH)
  extractor = Extractor()
  feature_dict = extractor.extract(nebula, return_dict=True)
  features = np.stack([np.array(list(v.values()))
                       for v in feature_dict.values()], axis=0)
  feature_names = list(list(feature_dict.values())[0].keys())

  target_labels = [TARGET]
  targets = [nebula.meta[pid][TARGET] for pid in nebula.labels]
  omix = Omix(features, targets, feature_names, nebula.labels, target_labels,
              data_name=f'OSA-N125-C6-P39-30s')

  set_target_collection_for_omix(omix, nebula)

  omix.save(OMIX_PATH)

omix.show_in_explorer()

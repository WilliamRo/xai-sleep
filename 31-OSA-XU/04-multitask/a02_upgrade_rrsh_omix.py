"""Upgrade existing omix file for RRSH-OSA dataset
"""
from numba.cuda import local

from pictor.xomics.omix import Omix

import os



# -----------------------------------------------------------------------------
# (1) Configuration
# -----------------------------------------------------------------------------
# (1.1) Paths
SOLUTION_DIR = '../../'
OMIX_DIR = os.path.join(SOLUTION_DIR, r'data/rrsh-osa/rrsh_osa_omix')

# (1.2) Specify feature set and target
FEATURE_SET_ID = 0
TARGET = ['OSA_M/MS', 'OSA_MM/S'][1]

# (*)
FEATURE_SET = ['macro', 'micro', 'hypno', 'macro_hypno'][FEATURE_SET_ID]

# -----------------------------------------------------------------------------
# (2) Generate and save .omix file if not exist
# -----------------------------------------------------------------------------
OMIX_PATH = None
if FEATURE_SET == 'macro':
  # (2.1) Macro-feature
  OMIX_FN = 'RRSH125_macro_D30.omix'
  OMIX_PATH = os.path.join(OMIX_DIR, OMIX_FN)

elif FEATURE_SET == 'micro':
  # (2.2) Micro-feature
  OMIX_FN = 'RRSH125_micro_D930.omix'
  OMIX_PATH = os.path.join(OMIX_DIR, OMIX_FN)

elif FEATURE_SET == 'hypno':
  # (2.3) Hypno-feature
  OMIX_FN = 'RRSH125_hypno_D4774.omix'
  OMIX_PATH = os.path.join(OMIX_DIR, OMIX_FN)

elif FEATURE_SET == 'macro_hypno':
  # (2.4) Macro-hypno-feature
  OMIX_FN = 'RRSH125_macro_hypno_D4804.omix'
  OMIX_PATH = os.path.join(OMIX_DIR, OMIX_FN)

  if not os.path.exists(OMIX_PATH):
    omix_macro = Omix.load(os.path.join(OMIX_DIR, 'RRSH125_macro_D30.omix'))
    target_collection = omix_macro.target_collection
    omix_macro.set_targets('AHI', False)

    omix_hypno = Omix.load(os.path.join(OMIX_DIR, 'RRSH125_hypno_D4774.omix'))
    omix_hypno.set_targets('AHI', False)

    omix = omix_hypno * omix_macro

    omix.put_into_pocket('target_collection', target_collection,
                         exclusive=False, local=True)

    omix.save(OMIX_PATH, verbose=True)

else: raise KeyError(f'Unknown feature set `{FEATURE_SET}`')

# -----------------------------------------------------------------------------
# (3) Upgrade omix file
# -----------------------------------------------------------------------------
# (3.1) Read omix
omix = Omix.load(OMIX_PATH)

# (3.2) Upgrade omix if not exist
if TARGET not in omix.target_collection:

  if TARGET == 'OSA_M/MS':
    threshold = 15
    target_labels = ['Mild', 'Mod/Sev']
  elif TARGET == 'OSA_MM/S':
    threshold = 30
    target_labels = ['Mil/Mod', 'Severe']
  else: raise KeyError(f'!! Unknown target `{TARGET}`')

  labels = [0 if ahi < threshold else 1
            for ahi in omix.target_collection['AHI'][0]]

  omix.add_to_target_collection(TARGET, labels, target_labels=target_labels)

  # Save
  omix.save(OMIX_PATH, verbose=True)

omix = omix.set_targets(TARGET, return_new_omix=True)
omix.data_name += f' ({TARGET})'

# (3.3) Explore omix using FeatureExplorer
omix.show_in_explorer()

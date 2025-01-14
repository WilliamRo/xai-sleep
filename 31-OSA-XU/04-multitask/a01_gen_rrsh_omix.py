"""This is for generating .omix files for
   (1) macro-features, D=30;
   (2) micro-features, D=930;
   (3) hypnomic features, D=4774;

This module should be executed in Windows systems.

Micro/Hypno omix files are generated by
   `xai-sleep\31-OSA-XU\02-regression\g03_OSA_hypnomix_gen.py`
"""
from pictor.xomics.omix import Omix

import os



# -----------------------------------------------------------------------------
# (1) Configuration
# -----------------------------------------------------------------------------
# (1.1) Paths
SOLUTION_DIR = '../../'
OMIX_DIR = os.path.join(SOLUTION_DIR, r'data/rrsh-osa/rrsh_osa_omix')
# NEB_FN = '125samples-6channels-39probes-30s.nebula'
# NEB_PATH = os.path.join(SOLUTION_DIR, f'data/rrsh-osa/rrsh_osa_neb/{NEB_FN}')

# (1.2) Specify feature set and target
FEATURE_SET_ID = 0
TARGET_ID = 11

# (*)
FEATURE_SET = ['macro', 'micro', 'hypno'][FEATURE_SET_ID]
TARGET = [
  'AHI',  # 0
  'age',  # 1
  'gender',  # 2: n=125 (38 female, 87 male)
  'MMSE',  # 3
  'cog_imp',  # 4: n=97 (82 negative, 15 positive)
  'dep',  # 5: n=87 (41 negative, 46 positive)
  'anx',  # 6: n=90 (60 negative, 30 positive)
  'som',  # 7: n=92 (65 negative, 27 positive)
  'PHQ9', # 8: depression score -> 5
  'GAD7', # 9: anxiety score -> 6
  'ESS', # 10: daytime sleepiness -> 7
  'OSA_MM/S', # 11: (73 Mil/Mod, 52 Severe) -> 0
][TARGET_ID]

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

else: raise KeyError(f'Unknown feature set `{FEATURE_SET}`')

# -----------------------------------------------------------------------------
# (3) In module testing
# -----------------------------------------------------------------------------
if __name__ == '__main__':
  # (3.1) Read omix
  omix = Omix.load(OMIX_PATH)

  # (3.2) Set target
  omix = omix.set_targets(TARGET, return_new_omix=True)
  omix.data_name += f' ({TARGET})'

  # (3.3) Explore omix using FeatureExplorer
  omix.show_in_explorer()

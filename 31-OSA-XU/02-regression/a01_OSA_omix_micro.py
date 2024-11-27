from pictor.xomics.omix import Omix

import os



# -----------------------------------------------------------------------------
# (1) Configuration
# -----------------------------------------------------------------------------
# (1.1) Nebula configuration
WORK_DIR = r'../data'  # contains cloud files

# (1.2) Set path
NEB_FN = f'125samples-6channels-39probes-30s.nebula'
NEB_PATH = os.path.join(WORK_DIR, NEB_FN)
OMIX_PATH = NEB_PATH.replace('.nebula', '.omix')

OVERWRITE = 0
TARGET = [
  'AHI',      # 0
  'age',      # 1
  'gender',   # 2
  'MMSE',     # 3
  'cog_imp',  # 4
  'dep',      # 5
  'anx',      # 6
  'som',      # 7
][1]
# -----------------------------------------------------------------------------
# (2) Load omix
# -----------------------------------------------------------------------------
assert os.path.exists(OMIX_PATH)

omix = Omix.load(OMIX_PATH)
omix = omix.set_targets(TARGET, return_new_omix=True)



if __name__ == '__main__':
  omix.show_in_explorer()


"""
Note: 
[AGE]
I. omix exploration pipeline
   (1) sf_pval 1000
   (2) sf_ucp 100 0.7 
       Note 0.7 works better than 0.9, yielding final MAE as low as 7.0 (age).
   (3) (optional) sf_pval N (N < 100)
   (4) ml eln/svr/lir
   
[GENDER]
(1) sf_pval 1000; (2) sf_ucp 100 0.7; (3) sf_pval 20; 
    - AUC=0.87
    
[MMSE] 82 (-) and 15 (+)
(1) sf_pval 1000; (2) sf_ucp 100 0.7; (3) sf_pval 50; 
    - ml lr, AUC=0.927
    
[dep] 41 (-) and 46 (+)
(1) sf_pval 1000; (2) sf_ucp 100 0.7; (3) sf_pval 50; 
    - ml lr, AUC=0.828
    
[dep] 60 (-) and 30 (+)
(1) sf_pval 1000; (2) sf_ucp 100 0.7; (3) sf_pval 50; 
    - ml lr, AUC=0.871
    
[som] 65 (-) and 27 (+)
(1) sf_pval 1000; (2) sf_ucp 100 0.7; (3) sf_pval 50; 
    - ml lr, AUC=0.878
"""

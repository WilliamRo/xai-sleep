"""
"""
import sys, os

SOLUTION_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
print(f'[RRSH-125] SOLUTION_DIR = `{SOLUTION_DIR}`')

PATH_LIST = ['31-OSA-XU', 'xai-kit', 'xai-kit/roma', 'xai-kit/pictor']

for p in PATH_LIST: sys.path.append(os.path.join(SOLUTION_DIR, p))

from pictor.xomics.omix import Omix
from pictor.xomics.evaluation.pipeline import Pipeline
from roma import io, console



# -----------------------------------------------------------------------------
# (1) Configuration
# -----------------------------------------------------------------------------
# (1.1) Fixed settings, do not edit
IN_LINUX = os.name != 'nt'
OMIX_DIR = os.path.join(SOLUTION_DIR, r'data/rrsh-osa/rrsh_osa_omix')
OMIX_FN = 'RRSH125_hypno_D4774.omix'
OMIX_PATH = os.path.join(OMIX_DIR, OMIX_FN)
PI_DIR = os.path.join(SOLUTION_DIR, r'data/rrsh-osa/rrsh_osa_pi')

# (1.2) Configure 1: target_id
if IN_LINUX:
  assert len(sys.argv) == 3, 'Please specify and target_id'
  TARGET_ID = int(sys.argv[1])
  PP = int(sys.argv[2])
else:
  TARGET_ID = 5
  PP = 3000
  # PI_DIR = r'\\192.168.5.100\xai-beta\xai-sleep\data\rrsh-osa\rrsh_osa_pi'

# (1.3) TODO: Configure 2: pipeline settings
n_splits = 5
M = 4
N = 4
ks = [100, 200]
ts = [0.7]

OVERWRITE = 0

# (*)
assert TARGET_ID in (2, 4, 5, 6, 7)
TARGET = [
  'AHI',  # 0
  'age',  # 1
  'gender',  # 2: n=125 (38 female, 87 male)
  'MMSE',  # 3
  'cog_imp',  # 4: n=97 (82 negative, 15 positive)
  'dep',  # 5: n=87 (41 negative, 46 positive)
  'anx',  # 6: n=90 (60 negative, 30 positive)
  'som',  # 7: n=92 (65 negative, 27 positive)
][TARGET_ID]

OMIX_PREFIX = OMIX_FN.split('.')[0]
k_str = ','.join([str(k) for k in ks])
t_str = ','.join([str(t) for t in ts])
n_str = '' if n_splits == 5 else f'_ns{n_splits}'
PI_FN = f'{OMIX_PREFIX}_{TARGET}_M{M}N{N}_k({k_str})_t({t_str}){n_str}_PP{PP}.pi'
PI_PATH = os.path.join(PI_DIR, PI_FN)

# -----------------------------------------------------------------------------
# (2) Fit pipeline
# -----------------------------------------------------------------------------
if OVERWRITE or not os.path.exists(PI_PATH):
  # if not IN_LINUX: assert False

  # (2.0) Load omix
  omix = Omix.load(OMIX_PATH)
  omix = omix.set_targets(TARGET, return_new_omix=True)
  omix.data_name = f'{OMIX_PREFIX}({TARGET})'

  # TODO: check feature number
  assert omix.n_features == 4774

  # For hypnomic features, select top 1000 features
  if PP < 4774: omix = omix.select_features('pval', k=PP)

  # (2.1) Initialize pipeline using macro omix
  pi = Pipeline(omix, ignore_warnings=1, save_models=1)

  # (2.2) Create subspaces
  for k, t in [(_k, _t) for _k in ks for _t in ts]: pi.create_sub_space(
    'ucp', k=k, threshold=t, repeats=M, nested=True, show_progress=1)

  # (2.3) Traverse all subspaces
  pi.fit_traverse_spaces('lr', repeats=N, nested=1, show_progress=1,
                         n_splits=n_splits)
  # pi.fit_traverse_spaces('svm', repeats=N, nested=1, show_progress=1)

  # (2.4) Save packages if required
  io.save_file(pi, PI_PATH, verbose=True)
else:
  pi = io.load_file(PI_PATH, verbose=True)
  console.show_info(f'omix.data_name = `{pi.omix.data_name}`')

# -----------------------------------------------------------------------------
# (3) Plot matrix
# -----------------------------------------------------------------------------
if not IN_LINUX: pi.plot_matrix(title=PI_FN)
else: pi.report()
"""
TARGET = 5, no pre 'pval'
-------------------------------------------------------------------------------
:: Pipeline Report
-------------------------------------------------------------------------------
:: Feature selection method: ucp-100-t0.7
...... Model: LogisticRegression
........ Avg(AUC) over 16 trials: 0.530, CI95% = [0.506, 0.553]
........ Avg(F1) over 16 trials: 0.521, CI95% = [0.502, 0.539]
:: Feature selection method: ucp-200-t0.7
...... Model: LogisticRegression
........ Avg(AUC) over 16 trials: 0.562, CI95% = [0.534, 0.590]
........ Avg(F1) over 16 trials: 0.539, CI95% = [0.512, 0.565]
-------------------------------------------------------------------------------
"""

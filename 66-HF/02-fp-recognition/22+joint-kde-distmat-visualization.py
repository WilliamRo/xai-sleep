from hypnomics.hypnoprints.extractor import Extractor

from x21_joint_kde_distmat_AUC import *
from hf.match_lab import MatchLab



# -----------------------------------------------------------------------------
# (1) Configuration
# -----------------------------------------------------------------------------
# Should be configured in x21_joint_kde_distmat_analysis.py

# -----------------------------------------------------------------------------
# (3) Visualize matrix
# -----------------------------------------------------------------------------
N = len(neb_1.labels)
matrices = []
labels = []

key_map = lambda k: f'{CK_MAP[k[0]]}-{PK_MAP[k[1]]}'
for i, key_i in enumerate(cpk_list):
  # key_i = (ck_i, pk_i)
  assert key_i in matrix_dict
  matrices.append(matrix_dict[key_i])
  labels.append(key_map(key_i))

  for j, key_j in enumerate(cpk_list[i + 1:]):
    joint_key = (key_i, key_j)
    if joint_key not in matrix_dict:
      console.warning(f'{joint_key} not found in matrix_dict')
      continue

    matrices.append(matrix_dict[joint_key])
    labels.append(f'{key_map(key_i)}x{key_map(key_j)}')


# TODO:
extractor = Extractor()
F1 = extractor.extract(neb_1, return_dict=True)
F2 = extractor.extract(neb_2, return_dict=True)

matlab = MatchLab(F1, F2, normalize=1, N=999,
                  neb_1=neb_1, neb_2=neb_2, nebula=nebula)
matlab.select_feature(min_ICC=0.5, verbose=1, set_C=1)

matlab.analyze(matrices=matrices, labels=labels, toolbar=1, omix=True)

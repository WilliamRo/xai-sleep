from hypnomics.freud.nebula import Nebula
from hf.model_helper import gen_dist_mat
from hf.sc_tools import get_dual_nebula, get_joint_key
from hf.sc_tools import CK_MAP, PK_MAP
from roma import console
from roma import finder
from x_dual_view import PAIRED_LABELS

import os
import numpy as np



# -----------------------------------------------------------------------------
# (1) Configuration
# -----------------------------------------------------------------------------
WORK_DIR = r'../data/sleepedfx_sc'
CHANNELS = [
  'EEG Fpz-Cz',
  'EEG Pz-Oz'
]

PROBE_KEYS = [
  'FREQ-20',
  'AMP-1',
  # 'GFREQ-35',
  # 'P-TOTAL',
  'RP-DELTA', 'RP-THETA', 'RP-ALPHA', 'RP-BETA',
]

# SG_LABELS = ['SC4001E', 'SC4002E']
SG_LABELS = finder.walk(WORK_DIR, type_filter='dir', return_basename=True)[:999]

# [ 2, 5, 10, 30, ]
TIME_RESOLUTION = 30

# NEB_FN = f'SC-{TIME_RESOLUTION}-KDE.nebula'
NEB_FN = f'SC-{TIME_RESOLUTION}-KDE-0730.nebula'
neb_file_path = os.path.join(WORK_DIR, NEB_FN)

CHNL_PROB_KEYS = [(ck, pk) for ck in CHANNELS for pk in PROBE_KEYS]

# -----------------------------------------------------------------------------
# (2) Load paired nebula
# -----------------------------------------------------------------------------
if os.path.exists(neb_file_path):
  nebula: Nebula = Nebula.load(neb_file_path)
else:
  raise FileNotFoundError(f'Nebula file not found: {neb_file_path}')

nebula.set_labels(PAIRED_LABELS)
neb_1, neb_2 = get_dual_nebula(nebula)

# -----------------------------------------------------------------------------
# (3) Generate matrix
# -----------------------------------------------------------------------------
N = len(neb_1.labels)
CHNL_PROB_PRODUCTS = [
  (ck1, pk1, ck2, pk2)
  for (ck1, pk1) in CHNL_PROB_KEYS for (ck2, pk2) in CHNL_PROB_KEYS
  if ck1 != ck2 or pk1 != pk2
]

matrix_dict, cpk_list = {}, []
for ck1, pk1, ck2, pk2 in CHNL_PROB_PRODUCTS:
  # Get joint_kde_dist_dict
  reverse_key = get_joint_key(ck2, pk2, ck1, pk1)
  if nebula.in_pocket(reverse_key): continue

  # 1-D KDE distance matrix
  key_1, key_2 = f'{ck1}-{pk1}-KDE-DISTS', f'{ck2}-{pk2}-KDE-DISTS'

  for (ck, pk) in [(ck1, pk1), (ck2, pk2)]:
    if (ck, pk) not in matrix_dict:
      mat_key_1D = f'{ck}-{pk}-KDE-DISTS'
      kde_dist_dict = nebula.get_from_pocket(mat_key_1D, key_should_exist=True)
      mat = gen_dist_mat(neb_1, neb_2, kde_dist_dict, mat_key_1D)
      matrix_dict[(ck, pk)] = mat
      cpk_list.append((ck, pk))

  # 2-D KDE distance matrix
  mat_key_2D = get_joint_key(ck1, pk1, ck2, pk2)
  kde_dist_dict = nebula.get_from_pocket(mat_key_2D, key_should_exist=False)
  if kde_dist_dict is not None:
    mat = gen_dist_mat(neb_1, neb_2, kde_dist_dict, mat_key_2D)
    matrix_dict[((ck1, pk1), (ck2, pk2))] = mat



if __name__ == '__main__':
  # -----------------------------------------------------------------------------
  # (4) Generate and show AUC matrix
  # -----------------------------------------------------------------------------
  from pictor.xomics.evaluation.roc import ROC

  def get_auc_ci(mat: np.ndarray):
    assert len(mat.shape) == 2

    features, targets, N = [], [], mat.shape[0]
    for i, j in [(i, j) for i in range(N) for j in range(N)]:
      features.append(mat[i, j])
      targets.append(0 if i == j else 1)

    roc = ROC(features, targets)
    # l, h = roc.calc_CI()
    l, h = 0, 0
    return roc.auc, l, h


  auc_matrix = np.zeros((len(cpk_list), len(cpk_list)))
  CIs = {}
  for i, key_i in enumerate(cpk_list):
    # key_i = (ck_i, pk_i)
    assert key_i in matrix_dict
    a, l, h = get_auc_ci(matrix_dict[key_i])
    auc_matrix[i, i] = a
    CIs[key_i] = (l, h)

    for j, key_j in enumerate(cpk_list[i+1:]):
      joint_key = (key_i, key_j)
      if joint_key not in matrix_dict:
        console.warning(f'{joint_key} not found in matrix_dict')
        continue

      a, l, h = get_auc_ci(matrix_dict[joint_key])
      auc_matrix[i, j + i + 1] = a
      CIs[joint_key] = (l, h)

  # -----------------------------------------------------------------------------
  # (5) Plot AUC matrix
  # -----------------------------------------------------------------------------
  from pictor.plotters.matrix_viewer import MatrixViewer

  row_labels = [f'{CK_MAP[ck]}-{PK_MAP[pk]}' for ck, pk in cpk_list]
  col_labels = row_labels
  cmap = ['Blues', 'Oranges'][0]
  MatrixViewer.show_matrices({'AUC': auc_matrix.T}, row_labels, col_labels,
                             (7, 7), cmap=cmap)







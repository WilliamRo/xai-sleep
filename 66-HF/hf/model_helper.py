import numpy as np



def gen_dist_mat(neb_1, neb_2, kde_dist_dict, mat_key):
  N = len(neb_1.labels)
  mat = np.zeros((N, N))
  for i, label_1 in enumerate(neb_1.labels):
    for j, label_2 in enumerate(neb_2.labels):
      k = (label_1, label_2)

      if k not in kde_dist_dict: raise KeyError(f'{k} not exist in {mat_key}')

      mat[i, j] = kde_dist_dict[k]
  return mat
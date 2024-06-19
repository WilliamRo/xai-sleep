from roma import Nomear

import numpy as np



class DistanceMatrixAgent(Nomear):

  def __init__(self, data1: np.ndarray, data2: np.ndarray):
    # Sanity check: data1 and data2 should be of same shape
    assert isinstance(data1, np.ndarray) and isinstance(data2, np.ndarray)
    assert data1.shape == data2.shape

    # Attributes
    self.data1, self.data2 = data1, data2

  # region: Properties

  @property
  def n_samples(self): return self.data1.shape[0]

  @property
  def n_features(self): return self.data1.shape[1]

  @Nomear.property()
  def distance_matrix_euclidean(self):
    N = self.n_samples
    matrix = np.zeros((N, N))
    for i in range(N):
      for j in range(N):
        matrix[i, j] = np.linalg.norm(self.data1[i] - self.data2[j])
    return matrix

  # endregion: Properties

  # region: Public Methods

  @staticmethod
  def calc_top_k_acc_based_on_distmat(mat: np.ndarray, k=1):
    N = mat.shape[0]
    indices = np.argsort(mat, axis=1)[:, :k]
    match = np.arange(N).reshape([-1, 1]) == indices
    match = np.max(match, axis=1)
    acc = np.mean(match)
    return acc

  def calc_top_k_accuracy(self, k=1):
    return self.calc_top_k_acc_based_on_distmat(
      self.distance_matrix_euclidean, k)

  # endregion: Public Methods



if __name__ == '__main__':
  data1 = np.arange(10).reshape(5, 2)
  data2 = data1

  dma = DistanceMatrixAgent(data1, data2)

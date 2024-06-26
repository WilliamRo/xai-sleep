from collections import OrderedDict
from roma import console
from roma import Nomear

import numpy as np



class MatchLab(Nomear):

  def __init__(self, F1: dict, F2: dict, N=999):
    self.F1, self.F2 = F1, F2
    self.V1, self.V2 = self._get_V(F1, N), self._get_V(F2, N)
    self.N, self.D = self.V1.shape[0], self.V1.shape[1]

  # region: Properties

  @Nomear.property()
  def delta_brick(self):
    return np.abs(self._get_brick(self.V1, 1) - self._get_brick(self.V2, 2))

  @property
  def feature_names(self):
    sample: dict = list(self.F1.values())[0]
    return list(sample.keys())

  # endregion: Properties

  # region: Private Methods

  def _get_V(self, F: dict, N):
    return np.stack([np.array(list(_dict.values()))
                     for _dict in F.values()], axis=0)[:N]

  def _get_brick(self, V: np.ndarray, night_id: int):
    """(euclidean) M = norm(B1 - B2, axis=2), M.shape = (N, N)
       M[i, j] = norm(F1[i] - F2[j])
    """
    N, D = V.shape

    if night_id == 1:
      B = V[:, np.newaxis, :]  # [N, 1, D]
    else:
      assert night_id == 2
      B = V[np.newaxis, :, :]  # [1, N, D]

    B = np.broadcast_to(B, (N, N, D))
    return B

  def _rank(self, matrices):
    # M.shape = [D, N, N]
    M = np.stack(matrices, axis=0)
    R = np.argsort(M, axis=2)
    D, N = M.shape[:2]
    position = np.zeros((D, N), dtype=int)
    for d in range(D):
      for n in range(N):
        indices = np.argwhere(R[d, n] == n)
        assert len(indices) == 1
        position[d, n] = indices[0]

    return position

  def _top_k_acc(self, matrices, k=1):
    P = self._rank(matrices)
    return np.mean(P < k, axis=1)

  def _rank_score(self, matrices):
    P = self._rank(matrices)
    return np.mean(P, axis=1)

  # endregion: Private Methods

  # region: Public Methods

  def analyze(self):
    from pictor import Pictor

    # (1)
    p = Pictor.image_viewer(f'MatchLab (N = {self.N})', figure_size=(7, 7))
    p.plotters[0].set('cmap', 'RdYlGn')
    p.plotters[0].set('color_bar', True)
    p.plotters[0].set('title', True)

    # (2)
    matrices = [self.delta_brick[..., d] for d in range(self.D)]
    labels = self.feature_names

    matrices.insert(0, np.linalg.norm(self.delta_brick, axis=-1))
    labels.insert(0, 'Norm')

    # (2.1)
    ACCs = self._top_k_acc(matrices, k=1)
    labels = [f'{lb}, Acc = {acc:.2f}' for lb, acc in zip(labels, ACCs)]

    # (2.2)
    scores = self._rank_score(matrices)
    labels = [f'{lb}, RS = {s:.1f}' for lb, s in zip(labels, scores)]

    # (3)
    p.objects = matrices
    p.labels = labels
    p.show()

  # endregion: Public Methods

  # region: MISC

  def test_brick(self):
    for i in range(self.N):
      for j in range(self.N):
        for d in range(self.D):
          assert self.delta_brick[i, j, d] == abs(self.V1[i][d] - self.V2[j][d])
    console.show_status('All tests passed!')

  # endregion: MISC

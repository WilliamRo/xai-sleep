from collections import OrderedDict
from hypnomics.freud.nebula import Nebula
from pictor.xomics.ml.ml_engine import FitPackage
from pictor.xomics.omix import Omix
from pictor.xomics.evaluation.pipeline import Pipeline
from roma import console
from roma import Nomear

import numpy as np



class MatchLab(Nomear):

  def __init__(self, F1: dict = None, F2: dict = None, N=999, normalize=1,
               neb_1=None, neb_2=None, nebula=None, kde_dist_dict=None):
    self.kde_dist_dict: OrderedDict = kde_dist_dict

    if not self.with_KDE:
      # Case I: Vectors are provided
      self.F1, self.F2 = F1, F2
      self.V1, self.V2 = self._get_V(F1, N), self._get_V(F2, N)

      self.nebula: Nebula = nebula
      self.neb_1: Nebula = neb_1
      self.neb_2: Nebula = neb_2

      if normalize:
        V = np.concatenate([self.V1, self.V2], axis=0)
        mu, sigma = np.mean(V, axis=0), np.std(V, axis=0)
        self.V1 = (self.V1 - mu) / (sigma + 1e-6)
        self.V2 = (self.V2 - mu) / (sigma + 1e-6)
    else:
      # Case II: KDEs are provided
      pass

    # Lab variables
    self.feature_coef = 1.0
    self.ICC_power = 3
    self._selected_feature_names = None

  # region: Properties

  @property
  def with_KDE(self): return getattr(self, 'kde_dist_dict', None) is not None

  @property
  def N(self):
    if self.with_KDE: return list(self.kde_dist_dict.values())[0].shape[0]
    return self.V1.shape[0]

  @property
  def D(self):
    if self.with_KDE: return len(self.kde_dist_dict)
    return self.V1.shape[1]

  @property
  def delta_brick(self):
    """DB.shape = [N, N, D]"""
    if not self.with_KDE:
      C = self.feature_coef
      if isinstance(C, np.ndarray):
        assert len(C) == self.D
        C = np.reshape(C, (1, 1, self.D))
      return np.abs(self._get_brick(self.V1, 1) - self._get_brick(self.V2, 2)) * C
    else:
      return np.stack(list(self.kde_dist_dict.values()), axis=-1)

  @property
  def feature_names(self):
    if not self.with_KDE:
      if self._selected_feature_names is not None:
        return self._selected_feature_names

      sample: dict = list(self.F1.values())[0]
      return list(sample.keys())
    else:
      return list(self.kde_dist_dict.keys())

  @property
  def distance_matrix(self):
    return np.linalg.norm(self.delta_brick, axis=-1)

  @property
  def sorted_ICC3_dict(self):
    return OrderedDict(sorted(self.ICC3_dict.items(), key=lambda x: x[1]))

  @Nomear.property()
  def ICC3_dict(self):
    import pingouin as pg
    import pandas as pd

    od = OrderedDict()
    for i, feature_name in enumerate(self.feature_names):
      values = np.concatenate([self.V1[:, i], self.V2[:, i]])
      data = pd.DataFrame(
        data={'Subject': list(range(self.N)) + list(range(self.N)),
              'Measurement': [1] * self.N + [2] * self.N,
              'Value': values})

      try:
        icc = pg.intraclass_corr(
          data, targets='Subject', raters='Measurement', ratings='Value')

        od[feature_name] = icc[icc['Type'] == 'ICC3']['ICC'].values[0]
      except:
        raise ValueError(f'Error in calculating ICC for {feature_name}')
        od[feature_name] = -1

    return od

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
    """Return rank, shape = [D, N]"""
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
    """Return top-k accuracy, shape = [D,]"""
    P = self._rank(matrices)
    return np.mean(P < k, axis=1)

  def _rank_score(self, matrices):
    P = self._rank(matrices)
    return np.mean(P, axis=1)

  # endregion: Private Methods

  # region: Public Methods

  def analyze(self, **kwargs):
    from pictor import Pictor

    # (0) Get distance matrix
    DM_KEY = 'distance_matrix'
    if DM_KEY in kwargs: dm = kwargs.pop(DM_KEY)
    else: dm = self.distance_matrix

    # (1)
    if 'matrices' in kwargs:
      matrices = kwargs.pop('matrices')
      labels = kwargs.pop('labels')
    else:
      matrices = [self.delta_brick[..., d] for d in range(self.D)]
      labels = [f'"{fn}"' for fn in self.feature_names]

    matrices.insert(0, dm)
    labels.insert(0, 'Distance Matrix')
    omix_labels = labels

    # (1.1)
    ACC1s = self._top_k_acc(matrices, k=1)
    ACC5s = self._top_k_acc(matrices, k=5)
    labels = [f'{lb}, ACC1/5 = {acc1:.2f}/{acc5:.2f}'
              for lb, acc1, acc5 in zip(labels, ACC1s, ACC5s)]

    # (1.2)
    scores = self._rank_score(matrices)
    labels = [f'{lb}, RS = {s:.1f}' for lb, s in zip(labels, scores)]

    # (2)
    assign_omix = kwargs.pop('omix', False)

    p = Pictor.image_viewer(
      f'MatchLab (N = {self.N})', figure_size=(7, 7), **kwargs)
    p.plotters[0].set('cmap', 'RdYlGn')
    p.plotters[0].set('color_bar', True)
    p.plotters[0].set('title', True)
    p.canvas._canvas.mpl_connect('button_press_event', self.on_click_show_dual)

    # (3)
    p.objects = matrices
    p.labels = labels

    # (4)
    if assign_omix:
      matrices = np.stack(matrices, axis=-1)
      features, targets = [], []
      for i, j in [(i, j) for i in range(self.N) for j in range(self.N)]:
        features.append(matrices[i, j])
        targets.append(0 if i != j else 1)

      features = np.stack(features, axis=0)
      omix = Omix(features, targets, feature_labels=omix_labels,
                  target_labels=['Not Match', 'Match'])
      p.plotters[0].omix = omix.show_in_explorer

    # (-1)
    p.show()

  def get_pair_omix(self, k=5, include_dm=False) -> Omix:
    N = self.N
    B = self.delta_brick
    argsort_DM = np.argsort(self.distance_matrix, axis=1)

    features, targets, sample_labels = [], [], []
    if self.with_KDE: feature_names = self.feature_names
    else: feature_names = [f'{name} (ICC = {self.ICC3_dict[name]:.3f})'
                           for name in self.feature_names]

    # Include distance matrix if required
    if include_dm:
      B = np.concatenate([self.distance_matrix[..., np.newaxis], B], axis=-1)
      feature_names.insert(0, 'MatchLab Distance')

    for n in range(N):
      # (1) Include match data
      features.append(B[n, n])
      targets.append(0)
      sample_labels.append(f'({n + 1}, {n + 1})')

      # (2) Include not-match data (top-K similar samples)
      other_indices = list(argsort_DM[n, :k + 1])
      if n in other_indices: other_indices.remove(n)
      else: other_indices = other_indices[:k]

      for i in other_indices:
        features.append(B[n, i])
        targets.append(1)
        sample_labels.append(f'({n + 1}, {i + 1})')

    # Wrap the data into Omix
    features = np.stack(features, axis=0)
    omix = Omix(features=features, targets=targets,
                feature_labels=feature_names, sample_labels=sample_labels,
                target_labels=['Match', 'Not Match'])
    return omix

  def fit_pipeline(self, omix: Omix, M=5, N=5, nested=False, **kwargs):
    # (0) Instantiate pipeline
    pi = Pipeline(omix, ignore_warnings=1, save_models=1)

    # (1) Create sub-spaces
    # (1.1) Create UCP-spaces
    if omix.n_features > 50:
      ks = kwargs.get('ks', [50, 100, 400])
      thresholds = kwargs.get('ts', [0.7, 0.9])
    else:
      ks = [20, 25]
      thresholds = [0.7, 0.8, 0.9]

    for k in ks:
      for threshold in thresholds:
        pi.create_sub_space('ucp', k=k, threshold=threshold,
                            repeats=M, nested=nested, show_progress=1)

    # (1.2) Candidate spaces
    # pi.create_sub_space('lasso', repeats=M, show_progress=1)
    # pi.create_sub_space('pca', n_components=k, repeats=M, show_progress=1)
    # pi.create_sub_space('mrmr', k=k, repeats=M, show_progress=1)
    # pi.create_sub_space('pval', k=10, repeats=M, nested=True, show_progress=1)

    # (3) Traverse through sub-spaces
    # (3.1) Logistic Regression
    lr_hp_space = None
    # lr_hp_space = {'C': [0.1, 1.0], 'lr_ratio': 0.0, 'penalty': 'elasticnet',
    #                'solver': 'saga'}
    pi.fit_traverse_spaces('lr', repeats=N, nested=nested,
                           show_progress=1, hp_space=lr_hp_space)

    # (3.2) Candidate models
    # pi.fit_traverse_spaces('svm', repeats=N, show_progress=1)
    # pi.fit_traverse_spaces('dt', repeats=N, show_progress=1)
    # pi.fit_traverse_spaces('rf', repeats=N, show_progress=1)
    # pi.fit_traverse_spaces('xgb', repeats=N, show_progress=1)

    return pi

  def dm_validate(self, pi: Pipeline, ranking=1, reducer=None):
    """TODO: """
    dr, pkg = pi.get_best_pipeline(rank=ranking, reducer=reducer)

    # Recover ordered probability
    y = np.zeros(shape=(self.N, self.N), dtype=np.float32)
    sample_labels = pkg['sample_labels']
    for di, sl in zip(pkg.probabilities[:, 1], sample_labels):
      # label = f'({i + 1}, {j + 1})'
      i_str, j_str = sl.split(', ')
      i = int(i_str[1:]) - 1
      j = int(j_str[:-1]) - 1
      y[i, j] = di

    # TODO: remove the routine below if things are all well
    # x = np.reshape(self.delta_brick, (self.N * self.N, self.D))
    ## y = dr.reduce_dimension(x)
    # y = pkg.predict_proba(x)
    # y = np.reshape(y[:, 1], (self.N, self.N))

    self.analyze(distance_matrix=y)

  # endregion: Public Methods

  # region: MISC

  def test_brick(self):
    for i in range(self.N):
      for j in range(self.N):
        for d in range(self.D):
          assert self.delta_brick[i, j, d] == abs(self.V1[i][d] - self.V2[j][d])
    console.show_status('All tests passed!')

  # endregion: MISC

  # region: Lab Methods

  def on_click_show_dual(self, event):
    try:
      i, j = int(np.round(event.ydata)), int(np.round(event.xdata))
      labels = [self.neb_1.labels[i], self.neb_2.labels[j]]
      if not event.dblclick: print(f'[{i}] {labels[0]}, [{j}] {labels[1]}')
    except: return

    if any([not event.dblclick, self.nebula is None,
            self.neb_1 is None, self.neb_2 is None]): return

    all_labels = self.nebula.labels
    self.nebula.set_labels(labels)

    from hypnomics.freud.telescopes.telescope import Telescope

    configs = {
      # 'xmin': 3, 'xmax': 9, 'ymin': 0, 'ymax': 200,
      'show_kde': 1,
      'show_scatter': 0,
      'show_vector': 0,
      # 'scatter_alpha': 0.05,
    }

    PK1 = 'FREQ-20'
    PK2 = 'AMP-1'
    self.nebula.dual_view(
      x_key=PK1, y_key=PK2, viewer_class=Telescope, **configs)

    self.nebula.set_labels(all_labels, check_sub_set=False)

  def ICC_analysis(self, ymax=None):
    import matplotlib.pyplot as plt

    sorted_od = self.sorted_ICC3_dict

    Y = np.arange(len(sorted_od))
    X = list(sorted_od.values())
    plt.plot(X, Y, 'o-')
    plt.xlabel('ICC3')
    plt.yticks(Y, list(sorted_od.keys()))
    plt.title('Feature Ranking')

    if ymax is not None: plt.ylim(Y[-1] - ymax, Y[-1] + 1)

    plt.grid(True)

    plt.tight_layout()
    plt.show()

  def select_feature(self, min_ICC=0.6, verbose=True, set_C=False):
    indices = []
    C = []
    _selected_feature_names = []
    for i, (feature_name, ICC) in enumerate(self.ICC3_dict.items()):
      if ICC < min_ICC: continue
      indices.append(i)
      C.append(ICC ** self.ICC_power)
      _selected_feature_names.append(feature_name)
      if verbose: console.show_status(
        f'Feature {feature_name} selected (ICC = {ICC:.3f})')

    indices = np.array(indices)
    self.V1 = self.V1[:, indices]
    self.V2 = self.V2[:, indices]

    self._selected_feature_names = _selected_feature_names

    if set_C: self.feature_coef = np.array(C)

  # endregion: Lab Methods

  # region: Efficacy Estimation

  def estimate_efficacy_v1(self, pi_key=None, plot_matrix=0, nested=False,
    fig_size=(5, 5), overwrite=0, M=2, N=2, omix=None, **kwargs):
    """Estimate the efficacy of the feature vector in the task of individual
    matching.
    """
    # (0) Set keys
    OMIX_KEY = 'PAIR_OMIX'

    # (1) Fit pipeline
    if isinstance(pi_key, Pipeline): pi = pi_key
    elif isinstance(pi_key, str) and self.in_pocket(pi_key) and not overwrite:
      pi: Pipeline = self.get_from_pocket(pi_key)
    else:
      # (1.2) Load or create omix
      if omix is not None:
        console.show_status('Omix has been provided')
      elif self.in_pocket(OMIX_KEY) and not overwrite:
        omix = self.get_from_pocket(OMIX_KEY)
        console.show_status('Pair-omix loaded.')
      else:
        omix = self.get_pair_omix(k=99999)
        self.put_into_pocket(OMIX_KEY, omix, local=True,
                             exclusive=not overwrite)
        console.show_status('Pair-omix created.')

      # (1.3) Fit pipeline
      omix = omix.duplicate()
      omix.report()

      # Filter features if required
      fbn = kwargs.get('fbn', None)
      if fbn is not None:
        console.show_status(f'Filtering features with `{fbn}`...')

        if fbn == 'ST':
          omix = omix.filter_by_name('IS_', include=False)
          omix = omix.filter_by_name('IC_', include=False)
        elif fbn == 'IS':
          omix = omix.filter_by_name('IS_', include=True)
        elif fbn == 'IC':
          omix = omix.filter_by_name('IC_', include=True)
        else: raise ValueError(f'Unknown fbn: {fbn}')

        omix.report()

      pi: Pipeline = self.fit_pipeline(omix, M=M, N=N, nested=nested, **kwargs)

      if isinstance(pi_key, str):
        self.put_into_pocket(pi_key, pi, local=True, exclusive=not overwrite)

    # (2) Report
    from pictor.xomics.stat_analyzers import calc_CI

    # This is for plotting matrix, see pipeline -> line 328 plot_matrix
    matrices = OrderedDict()
    value_dict = OrderedDict()

    def _report_mu_CI95(values, key, n_pkg):
      mu = np.mean(values)
      CI1, CI2 = calc_CI(values, alpha=0.95, key=key)
      info = f'Avg({key}) over {n_pkg} trials: {mu:.3f}'
      info += f', CI95% = [{CI1:.3f}, {CI2:.3f}]'
      console.supplement(info, level=4)

    console.section('Pipeline Report')
    row_labels, col_labels, matrix_dict = pi.get_pkg_matrix()

    # Initialize stuff for each metric
    for key in ('AUC', 'F1', 'Top1-Acc', 'Top5-Acc', 'TAR % FAR = 1%'):
      matrices[key] = np.zeros((len(row_labels), len(col_labels)))
      value_dict[key] = OrderedDict()

      for r, sf_key in enumerate(row_labels):
        value_dict[key][r] = OrderedDict()

    for r, sf_key in enumerate(row_labels):
      console.show_info(f'Feature selection method: {sf_key}')
      for c, ml_key in enumerate(col_labels):
        console.supplement(f'Model: {ml_key}', level=3)
        pkg_list = matrix_dict[(sf_key, ml_key)]
        n_pkg = len(pkg_list)

        # (3.1) Report AUC & F1
        key = 'AUC'
        values = [p[key] for p in pkg_list]
        matrices[key][r, c] = np.mean(values)
        value_dict[key][r][c] = values
        _report_mu_CI95(values, key, n_pkg)

        key = 'F1'
        values = [p[key] for p in pkg_list]
        matrices[key][r, c] = np.mean(values)
        value_dict[key][r][c] = values
        _report_mu_CI95(values, key, n_pkg)

        # (3.2) Report Top-k acc
        values_top1, values_top5 = [], []
        for pkg in pkg_list:
          # Re-construct distance matrix
          y = np.zeros(shape=(self.N, self.N), dtype=np.float32)
          sample_labels = pkg['sample_labels']
          for di, sl in zip(pkg.probabilities[:, 1], sample_labels):
            # label = f'({i + 1}, {j + 1})'
            i_str, j_str = sl.split(', ')
            i = int(i_str[1:]) - 1
            j = int(j_str[:-1]) - 1
            y[i, j] = di

          values_top1.append(self._top_k_acc([y], k=1)[0])
          values_top5.append(self._top_k_acc([y], k=5)[0])

        key = 'Top1-Acc'
        values = values_top1
        matrices[key][r, c] = np.mean(values)
        value_dict[key][r][c] = values
        _report_mu_CI95(values_top1, key, n_pkg)

        key = 'Top5-Acc'
        values = values_top5
        matrices[key][r, c] = np.mean(values)
        value_dict[key][r][c] = values
        _report_mu_CI95(values_top5, key, n_pkg)

        # (3.3) Report TAR @ FAR
        values = []
        for pkg in pkg_list:
          assert isinstance(pkg, FitPackage)
          p = pkg.probabilities[:, 1]
          y = pkg.targets

          tar = self.calculate_TAR_at_FAR(p, y, target_FAR=0.01)
          values.append(tar)

        key = 'TAR % FAR = 1%'
        matrices[key][r, c] = np.mean(values)
        value_dict[key][r][c] = values
        _report_mu_CI95(values, key, n_pkg)

    print('-' * 79)

    # (-1) Show matrix if required
    if not plot_matrix: return pi

    from pictor.plotters.matrix_viewer import MatrixViewer

    MatrixViewer.show_matrices(matrices, row_labels, col_labels, fig_size,
                               values=value_dict, cmap='Blues', n_digit=3)

    return pi


  @staticmethod
  def calculate_TAR_at_FAR(d, y, target_FAR=0.001):
    """Calculate TAR at specified FAR

      Args:
          d: array of distances
          y: array of labels (1 for impostor, 0 for genuine)
          target_FAR: target FAR value (default: 0.001 for 0.1%)
      Returns:
          TAR value at the specified FAR
      """
    # Split distances into genuine and impostor
    d_impostor = d[y == 1]
    d_genuine = d[y == 0]

    # Sort impostor distances
    sorted_impostor = np.sort(d_impostor)

    # Find threshold at target FAR
    threshold_idx = int(np.ceil(len(d_impostor) * target_FAR)) - 1
    threshold = sorted_impostor[threshold_idx]

    # Calculate TAR
    TAR = np.sum(d_genuine <= threshold) / len(d_genuine)

    return TAR

  # endregion: Efficacy Estimation

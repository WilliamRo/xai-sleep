from pictor.xomics.omix import Omix
from pictor.xomics.evaluation.pipeline import Pipeline, FitPackage
from roma import console, io

import numpy as np



# -----------------------------------------------------------------------------
# (1) Read Omix
# -----------------------------------------------------------------------------
file_path = r'./data/SC-age-153x375-30s.omix'
omix = Omix.load(file_path)

omix = omix.select_features('pval', k=20)
# -----------------------------------------------------------------------------
# (2) Feature selection
# -----------------------------------------------------------------------------
from z01_evaluation import calc_mae
X, y = omix.features, omix.targets

if 0:
  from pictor.xomics.ml.elastic_net import ElasticNet

  model = ElasticNet(ignore_warnings=True)
  estimator = model.fit(omix, hp={'alpha': 1.0, 'l1_ratio': 0.0})
  y_pred = estimator.predict(X)

  mae = np.mean(np.abs(y_pred - y))
  print('ElasticNet MAE =', mae)
  print('score =', estimator.score(X, y))
  # pkg = model.fit_k_fold(omix, verbose=1)
  # pkg.report()

  console.section('TFENet')
  from hf.models.enet import TFENet

  model = TFENet(ignore_warnings=True)
  estimator = model.fit(omix, hp={
    'alpha': 1.0, 'l1_ratio': 0.0,
    'lr': 0.1,
  })
  y_pred = estimator.predict(omix.features)
  mae = np.mean(np.abs(y_pred - y))
  print('tfENet MAE =', mae)
  print('score =', estimator.score(X, y))

  exit()

if 0:
  #k (1)
  from pictor.xomics.ml.elastic_net import ElasticNet

  model = ElasticNet(ignore_warnings=True)
  pkg = model.fit_k_fold(omix, verbose=1)
  pkg.report()

  # (2)
  from hf.models.enet import TFENet

  model = TFENet(ignore_warnings=True)
  pkg = model.fit_k_fold(omix, verbose=1)
  pkg.report()

  exit()


# (1)
from pictor.xomics.ml.elastic_net import ElasticNet

model = ElasticNet(ignore_warnings=True)
pkg = model.fit_k_fold(omix, verbose=1)
pkg.report()

# (2)
from hf.models.pos_linear import SoftLinear

model = SoftLinear(ignore_warnings=True)
pkg = model.fit_k_fold(omix, verbose=1)
pkg.report()

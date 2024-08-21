from pictor.xomics.ml.linear_regression import LinearRegression
from pictor.xomics.omix import Omix
from pictor.xomics.evaluation.pipeline import Pipeline, FitPackage
from roma import console, io

import numpy as np



# -----------------------------------------------------------------------------
# (1) Read Omix
# -----------------------------------------------------------------------------
file_path = r'./data/SC-age-153x375-30s.omix'
omix = Omix.load(file_path)

omix = omix.select_features('pval', k=5)
X, y = omix.features, omix.targets
# -----------------------------------------------------------------------------
# (2) sklearn.LinearRegression fit
# -----------------------------------------------------------------------------
console.section('sklearn.LinearRegression fit')

# (2.1) LiR with intercept
model = LinearRegression(ignore_warnings=True)
estimator = model.fit(omix, hp={'fit_intercept': True})
console.show_info('LiR with intercept:')
console.supplement(f'W = {estimator.coef_}', level=2)
console.supplement(f'b = {estimator.intercept_}', level=2)

y_pred = estimator.predict(X)
mae = np.mean(np.abs(y_pred - y))
console.supplement(f'LiR MAE = {mae:.2f}', level=2)

# (2.2) LiR without intercept
model = LinearRegression(ignore_warnings=True)
estimator = model.fit(omix, hp={'fit_intercept': False})
console.show_info('LiR without intercept:')
console.supplement(f'W = {estimator.coef_}', level=2)

y_pred = estimator.predict(X)
mae = np.mean(np.abs(y_pred - y))
console.supplement(f'LiR MAE = {mae:.2f}', level=2)
# -----------------------------------------------------------------------------
# (3) Gradient descent fit
# -----------------------------------------------------------------------------
console.section('Gradient descent fit')

# (3.0) Configure
lr = 0.1
max_iters = 1000
tol = 1e-3
print_cycle = 50
patience = 10

# (3.1) Initialize variables
# W = np.zeros(shape=[X.shape[1], 1])
W = np.random.normal(size=[X.shape[1], 1])
b = np.zeros(shape=[1])
p = 10

# (3.2) Gradient descent
best_mae = np.inf
N, D = X.shape
best_W, best_b = W, b
for i in range(max_iters):
  if p < 0: break

  y_pred = np.matmul(X, W) + b
  error = y_pred - y.reshape(-1, 1)
  assert error.shape == (N, 1)

  # X.T.shape = [D, N], error.shape = [N, 1] -> dW.shape = [D, 1]
  dW = 2 * np.matmul(X.T, error) / N
  db = 2 * np.sum(error) / N

  W -= lr * dW
  b -= lr * db

  mae = np.mean(np.abs(error))
  if mae < best_mae - tol:
    best_W, best_b, best_mae = W, b, mae
    p = patience
  else: p -= 1

  if i % print_cycle == 0:
    console.show_status(f'Iter-{i+1}: MAE = {mae:.2f}')

# (3.3) Report final results
console.show_info('Gradient descent result:')
console.supplement(f'W = {best_W.ravel()}', level=2)
console.supplement(f'b = {best_b[0]}', level=2)
console.supplement(f'MAE = {best_mae:.2f}', level=2)

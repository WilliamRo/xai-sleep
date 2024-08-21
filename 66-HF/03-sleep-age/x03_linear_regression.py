from pictor.xomics.omix import Omix
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from z01_evaluation import plot_age_est, calc_mae

import numpy as np



# -----------------------------------------------------------------------------
# (1) Read Omix
# -----------------------------------------------------------------------------
file_path = r'./data/SC-age-153x375-30s.omix'
omix = Omix.load(file_path)

# -----------------------------------------------------------------------------
# (2) Feature selection
# -----------------------------------------------------------------------------
for D in range(5, 50):
  indices = np.argsort([r.f_pvalue for r in omix.OLS_reports])
  indices = indices[:D]
  omix_D = omix.get_sub_space(indices, start_from_1=False)

  # -----------------------------------------------------------------------------
  # (3) Regression
  # -----------------------------------------------------------------------------
  X, y = omix_D.features, omix_D.targets

  # Split data
  test_size = 0.2
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

  reg = LinearRegression().fit(X_train, y_train)
  pred = reg.predict(X_test)

  mu, sigma = calc_mae(y_test, pred)
  title = f'MAE = {mu:.2f} ± {sigma:.2f} years'
  print(f'D = {D}, {title}')


from pictor.xomics.omix import Omix
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from z01_evaluation import plot_age_est

import numpy as np



# -----------------------------------------------------------------------------
# (1) Read Omix
# -----------------------------------------------------------------------------
file_path = r'./data/SC-age-153x375-30s.omix'
omix = Omix.load(file_path)

# omix.show_in_explorer()
# exit()
# -----------------------------------------------------------------------------
# (2) Feature selection
# -----------------------------------------------------------------------------
D = 30
indices = np.argsort([r.f_pvalue for r in omix.OLS_reports])
indices = indices[:D]
omix = omix.get_sub_space(indices, start_from_1=False)

# omix.show_in_explorer()
# exit()

# -----------------------------------------------------------------------------
# (3) Regression
# -----------------------------------------------------------------------------
X, y = omix.features, omix.targets

# Split data
test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

# reg = LinearRegression().fit(X, y)
# pred = reg.predict(X)
# plot_age_est(y, pred)

reg = LinearRegression().fit(X_train, y_train)
pred = reg.predict(X_test)
pred_train = reg.predict(X_train)

plot_age_est(y_test, pred, train_data=(y_train, pred_train))


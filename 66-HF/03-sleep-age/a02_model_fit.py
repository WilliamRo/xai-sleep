from pictor.xomics.omix import Omix



# -----------------------------------------------------------------------------
# (1) Read Omix
# -----------------------------------------------------------------------------
file_path = r'./data/SC-age-153x375-30s.omix'
omix = Omix.load(file_path)

# -----------------------------------------------------------------------------
# (2) Recuce dimension
# -----------------------------------------------------------------------------
omix = omix.select_features('pval', k=20)

# -----------------------------------------------------------------------------
# (3) Fit model
# -----------------------------------------------------------------------------
from pictor.xomics.ml.linear_regression import LinearRegression

model = LinearRegression(ignore_warnings=True)
pkg = model.fit_k_fold(omix, verbose=1)
pkg.report()

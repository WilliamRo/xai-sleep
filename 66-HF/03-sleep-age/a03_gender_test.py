from pictor.xomics.omix import Omix



# -----------------------------------------------------------------------------
# (1) Read Omix
# -----------------------------------------------------------------------------
file_path = r'./data/SC-age-153x375-30s.omix'
omix = Omix.load(file_path)

file_path = r'./data/SC-age-71x375-30s-male.omix'
omix_male = Omix.load(file_path)

file_path = r'./data/SC-age-82x375-30s-female.omix'
omix_female = Omix.load(file_path)
# -----------------------------------------------------------------------------
# (2) Recuce dimension
# -----------------------------------------------------------------------------
omix = omix.select_features('pval', k=20)

omix_male = omix.select_samples(omix_male.sample_labels)
omix_female = omix.select_samples(omix_female.sample_labels)
# -----------------------------------------------------------------------------
# (3) Fit model
# -----------------------------------------------------------------------------
from pictor.xomics.ml.linear_regression import LinearRegression
from pictor.xomics.ml.elastic_net import ElasticNet

ModelClass = ElasticNet

# (1) All
model = ModelClass(ignore_warnings=True)
pkg = model.fit_k_fold(omix, verbose=1)
pkg.report()

# (2) Male
model = ModelClass(ignore_warnings=True)
pkg = model.fit_k_fold(omix_male, verbose=1)
pkg.report()

# (3) Female
model = ModelClass(ignore_warnings=True)
pkg = model.fit_k_fold(omix_female, verbose=1)
pkg.report()


from roma import console
from pictor.xomics.omix import Omix
from pictor.xomics.evaluation.pipeline import Pipeline, FitPackage

import numpy as np



# -----------------------------------------------------------------------------
# Load data (Do not modify codes in this section)
# -----------------------------------------------------------------------------
console.section('Data preparation')

# data_path = r'../data/radiomics-111x851.omix'
data_path = r'../data/radiomics-67x851.omix'

omix = Omix.load(data_path)

omix_internal, omix_external = omix.split(1, 1, shuffle=False)
omix_internal.report()

# omix = omix.duplicate(target_labels=['Negative', 'Positive'])
omix.show_in_explorer()
exit()

X = omix_internal.features
y = omix_internal.targets
# -----------------------------------------------------------------------------
# TODO: Import packages and construct machine learning workflow
# -----------------------------------------------------------------------------
console.section('Learning on internal data')

# TODO: attention, this reference is sub-optimal

pi = Pipeline(omix_internal, ignore_warnings=1, save_models=1)
M = 2
pi.create_sub_space('lasso', repeats=M, show_progress=1)
pi.create_sub_space('pca', k=20, repeats=M, show_progress=1)
N = 2
pi.fit_traverse_spaces('lr', repeats=N, nested=1, show_progress=1, verbose=0)
pi.fit_traverse_spaces('xgb', repeats=N, nested=1, show_progress=1, verbose=0)
pi.report()

pi.plot_matrix()

# Wrap your predictor into a function
def predict_probabilities(_X):
  assert len(_X.shape) == 2

  dr, pkg = pi.get_best_pipeline()

  dummy_targets = [0] * len(_X)
  omix = Omix(_X, targets=dummy_targets)
  omix_reduced = dr.reduce_dimension(omix)
  prob = pkg.predict_proba(omix_reduced.features)

  return prob

# -----------------------------------------------------------------------------
# Validating model on external data
# -----------------------------------------------------------------------------
console.section('Validating on external data')

prob = predict_probabilities(omix_external.features)
predictions = np.argmax(prob, axis=1)
package = FitPackage.pack(predictions, prob, omix_external)
package.report()

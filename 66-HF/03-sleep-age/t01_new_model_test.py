from pictor.xomics.omix import Omix
from pictor.xomics.evaluation.pipeline import Pipeline, FitPackage
from roma import io

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
from hf.models.pos_linear import SoftLinear

pi = Pipeline(omix, ignore_warnings=1, save_models=1)
M = 2
pi.create_sub_space('pval', k=20, repeats=M, show_progress=1, nested=1)
# pi.create_sub_space('pca', k=20, repeats=M, show_progress=1, nested=1)
# pi.create_sub_space('lasso', repeats=M, show_progress=1, nested=1)
N = 2
pi.fit_traverse_spaces(SoftLinear, repeats=N, nested=1, show_progress=1,
                       verbose=0)
# pi.fit_traverse_spaces('lir', repeats=N, nested=1, show_progress=1, verbose=0)
# pi.fit_traverse_spaces('eln', repeats=N, nested=1, show_progress=1, verbose=0)
# pi.fit_traverse_spaces('svr', repeats=N, nested=1, show_progress=1, verbose=0)

pi.report()
# pi.plot_matrix()

# ---------------------------------------------------------------------------
# Save probability predictor
# ---------------------------------------------------------------------------
# omix_save_path = r'./data/0813_age_01.omix'
# omix.save(omix_save_path, verbose=True)
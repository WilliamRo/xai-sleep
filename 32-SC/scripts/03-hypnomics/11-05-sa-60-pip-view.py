from hypnomics.hypnoprints.hp_extractor import extract_hypnoprints_from_hypnocloud
from pictor.xomics.omix import Omix
from collections import OrderedDict
from roma import io, console

import os
import pandas as pd
import numpy as np



# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
data_dir = r'../../features/'
file_name = r'sleep-age-60.omix'
save_path = os.path.join(data_dir, file_name)

# -----------------------------------------------------------------------------
# Load cloud and excel
# -----------------------------------------------------------------------------
omix = Omix.load(save_path)
# omix.show_in_explorer()

# -----------------------------------------------------------------------------
# Pipeline
# -----------------------------------------------------------------------------
from pictor.xomics.evaluation.pipeline import Pipeline

pi = Pipeline(omix, ignore_warnings=1, save_models=0)
M = 5
pi.create_sub_space('lasso', repeats=M, show_progress=1)
pi.create_sub_space('*', repeats=M, show_progress=1)

N = 5
pi.fit_traverse_spaces('lr', repeats=N, show_progress=1)
pi.fit_traverse_spaces('svm', repeats=N, show_progress=1)
pi.fit_traverse_spaces('dt', repeats=N, show_progress=1)
pi.fit_traverse_spaces('rf', repeats=N, show_progress=1)
pi.fit_traverse_spaces('xgb', repeats=N, show_progress=1)

pi.report()

omix.save(os.path.join(data_dir, '20240610-sa-60.omix'), verbose=True)

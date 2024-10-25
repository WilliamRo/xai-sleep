from pictor.xomics.omix import Omix
from pictor.xomics.evaluation.pipeline import Pipeline

import os



# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------
data_dir = r'../../../data/rrsh-osa'

file_name = [
  '20240530.omix',
  '20240530-v1.omix',
][1]

# -----------------------------------------------------------------------------
# Combo
# -----------------------------------------------------------------------------
omix = Omix.load(os.path.join(data_dir, file_name))

pi = Pipeline(omix, ignore_warnings=1, save_models=0)

pi.report()
pi.plot_matrix()

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
file_names = [
'20240610-sa-60.omix',
]
file_path = os.path.join(data_dir, file_names[0])

# -----------------------------------------------------------------------------
# Load cloud and excel
# -----------------------------------------------------------------------------
omix = Omix.load(file_path)
# omix.show_in_explorer()

# -----------------------------------------------------------------------------
# Pipeline
# -----------------------------------------------------------------------------
from pictor.xomics.evaluation.pipeline import Pipeline

pi = Pipeline(omix, ignore_warnings=1, save_models=0)

pi.report()
pi.plot_matrix()

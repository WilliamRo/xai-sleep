from collections import OrderedDict
from hypnomics.hypnoprints import extract_hypnocloud_from_signal_group
from hypnomics.hypnoprints.hp_extractor import extract_hypnoprints_from_hypnocloud
from roma import finder
from roma import io, console

from sc.fp_viewer import FPViewer
from pictor.xomics.omix import Omix

import pandas as pd
import numpy as np
import os



# -----------------------------------------------------------------------------
# Configs
# -----------------------------------------------------------------------------
N = 125
reso = 30
NC = 2

# -----------------------------------------------------------------------------
# Load clouds (OrderedDict)
# -----------------------------------------------------------------------------
cloud_data_dir = r'../../features/'
cloud_file_name = f'OSA-{N}pts-{NC}chs-{reso}s.clouds'
clouds = io.load_file(os.path.join(cloud_data_dir, cloud_file_name), verbose=True)

# -----------------------------------------------------------------------------
# Load excel, find targets
# -----------------------------------------------------------------------------
XLSX_PATH = r'../../../data/rrsh-osa/OSA-xu.xlsx'
df = pd.read_excel(XLSX_PATH)

feature_names = None
hypn_dict = OrderedDict()
for pid, cloud in clouds.items():
  if feature_names is None:
    x_dict = extract_hypnoprints_from_hypnocloud(cloud, return_dict=True)
    hypn_dict[pid] = np.array(list(x_dict.values()))
    feature_names = list(x_dict.keys())
  else: hypn_dict[pid]= extract_hypnoprints_from_hypnocloud(cloud)

features = np.stack(list(hypn_dict.values()))

targets = [
  df.loc[df['序号'] == int(pid), '分组'].values[0] - 1
  for pid in clouds.keys()
]
targets = [0 if t == 0 else 1 for t in targets]
targets = np.array(targets)
target_labels = ['Group 1', 'Group 2&3']

# -----------------------------------------------------------------------------
# Wrap and show
# -----------------------------------------------------------------------------
sample_labels = None
omix = Omix(features, targets, feature_names, sample_labels, target_labels,
            data_name=f'OSA-{NC}chs-{reso}s')

# -----------------------------------------------------------------------------
# Combo
# -----------------------------------------------------------------------------
from pictor.xomics.evaluation.pipeline import Pipeline

data_dir = r'../../../data/rrsh-osa'
pi = Pipeline(omix, ignore_warnings=1, save_models=0)

M = 10
pi.create_sub_space('lasso', repeats=M, show_progress=1)
k = pi.lasso_dim_median
pi.create_sub_space('sig', n_components=k, repeats=M, show_progress=1)
pi.create_sub_space('pca', n_components=k, repeats=M, show_progress=1)
pi.create_sub_space('mrmr', k=k, repeats=M, show_progress=1)

N = 10
pi.fit_traverse_spaces('lr', repeats=N, show_progress=1)
pi.fit_traverse_spaces('svm', repeats=N, show_progress=1)
pi.fit_traverse_spaces('dt', repeats=N, show_progress=1)
pi.fit_traverse_spaces('rf', repeats=N, show_progress=1)
pi.fit_traverse_spaces('xgb', repeats=N, show_progress=1)

pi.report()

omix.save(os.path.join(data_dir, '20240530-v1.omix'), verbose=True)
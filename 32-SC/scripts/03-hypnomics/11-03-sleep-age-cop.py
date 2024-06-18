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
# Configs
N = 399
R = 30
overwrite = False
N = min(N, 153)

save_to_dir = r'../../features/'
cloud_file_name = f'SC-pt{N}-C2-dt{R}.clouds'
save_path = os.path.join(save_to_dir, cloud_file_name)

channels = ['EEG Fpz-Cz', 'EEG Pz-Oz']

XLSX_PATH = r'../../../data/sleep-edf-database-expanded-1.0.0/SC-subjects.xls'

# -----------------------------------------------------------------------------
# Load cloud and excel
# -----------------------------------------------------------------------------
clouds = io.load_file(save_path, verbose=True)
df = pd.read_excel(XLSX_PATH)

# -----------------------------------------------------------------------------
# Wrap
# -----------------------------------------------------------------------------
feature_names = None
hypn_dict = OrderedDict()
for pid, cloud in clouds.items():
  if feature_names is None:
    x_dict = extract_hypnoprints_from_hypnocloud(cloud, return_dict=True)
    hypn_dict[pid] = np.array(list(x_dict.values()))
    feature_names = list(x_dict.keys())
  else: hypn_dict[pid] = extract_hypnoprints_from_hypnocloud(cloud)

features = np.stack(list(hypn_dict.values()))

T = 60

targets = [
  df.loc[df['subject'] == int(pid[3:5]), 'age'].values[0]
  for pid in clouds.keys()
]

omix = Omix(features, targets, feature_names, None, ['Age'])
print(np.median(omix.targets))
# omix.show_in_explorer()






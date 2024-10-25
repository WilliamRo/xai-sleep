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
# targets = [0 if t == 0 else 1 for t in targets]
targets = np.array(targets)
target_labels = ['Group 1', 'Group 2', 'Group 3']

# -----------------------------------------------------------------------------
# Wrap and show
# -----------------------------------------------------------------------------
omix = Omix(features, targets, feature_names, target_labels,
            data_name=f'OSA-{NC}chs-{reso}s')

omix.show_in_explorer()


from freud.talos_utils.slp_set import SleepSet
from roma import console

import os
import time



data_root = r'E:\xai-sleep\data\sleepedf'
edf_path = r'SC4001EC-Hypnogram.edf'

file_path = os.path.join(data_root, edf_path)



from mne import read_annotations

raw_anno = read_annotations(file_path)

print()

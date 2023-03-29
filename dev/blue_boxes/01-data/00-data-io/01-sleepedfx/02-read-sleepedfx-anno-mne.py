from freud.talos_utils.slp_set import SleepSet
from roma import console

import os



data_root = r'../../../../../data/sleepedfx'
edf_path = r'SC4001EC-Hypnogram.edf'
file_path = os.path.join(data_root, edf_path)

anno = SleepSet.read_annotations_mne(file_path)
console.show_info(f'Annotation (num={len(anno.annotations)})')
console.supplement(f'Labels: {anno.labels}', level=2)

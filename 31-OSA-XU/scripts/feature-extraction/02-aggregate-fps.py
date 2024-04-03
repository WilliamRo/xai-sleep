from ew_viewer import EWViewer
from roma import console
from tframe.utils.file_tools.io_utils import load

import matplotlib.pyplot as plt
import os
import pandas as pd
import warnings



# Ignore all warnings
warnings.filterwarnings("ignore")

# Configuration
FP_PATH = r'../../features/0203-N125-ch(all)-F(20)-A(128).fp'
EXPORT_PATH = r'../../features/sub-data-02'
AGG_PATH = r'../../features/sub-data-02/agg'
XLSX_PATH = r'../../../data/rrsh-osa/OSA-wm.xlsx'

OVERWRITE = 1

HEAD = []
HEAD.append(('序号', r'PID: {}'))
HEAD.append(('分组', 'Group: {:.0f}'))
HEAD.append(('年龄', 'Age: {:.0f}'))
HEAD.append(('性别，男1，女2', 'Gender: {:.0f}'))
HEAD.append(('BMI', 'BMI: {:.1f}'))
HEAD.append(('AHI', 'AHI: {:.1f}'))
HEAD.append(('REM期AHI', 'REM-AHI: {:.1f}'))
HEAD.append(('s_MMSE', 'MMSE: {:.0f}'))
HEAD.append(('s_PHQ9', 'PHQ9: {:.0f}'))
HEAD.append(('s_GAD7', 'GAD7: {:.0f}'))
HEAD.append(('s_ESS', 'ESS: {:.0f}'))

# Get sg list and channel list
results = load(FP_PATH)
meta = results['meta']

ew = EWViewer(walker_results=results)

sg_label_list, channel_list, _ = ew.data['meta']

# Load excel
df = pd.read_excel(XLSX_PATH)

N, i = len(sg_label_list), 0
for sg_label in sg_label_list:
  console.show_status(f'Aggregating PID-{sg_label} ...')
  console.print_progress(i, N)

  agg_path = os.path.join(AGG_PATH, f'{int(sg_label):03d}.png')
  if os.path.exists(agg_path) and not OVERWRITE:
    N -= 1
    continue

  file_paths = [os.path.join(EXPORT_PATH, f'{sg_label},{ch}.png')
                for ch in channel_list]
  if not os.path.exists(file_paths[-1]): break

  x = [plt.imread(fp) for fp in file_paths]
  images = [x[0], x[2], x[3], x[4],
            x[1], x[-3], x[-2], x[-1]]
  fig = plt.figure(figsize=(22, 10))
  for j, im in enumerate(images):
    plt.subplot(2, 4, j + 1)
    plt.axis('off')
    plt.imshow(im)

  # Set suptitle based on excel
  indices = df[df['序号'] == int(sg_label)].index.tolist()
  assert len(indices) == 1
  index = indices[0]
  row = df.iloc[index]
  title = ' | '.join([fmt.format(row[k]) for k, fmt in HEAD])
  plt.suptitle(title)

  # Save image
  plt.tight_layout()
  plt.savefig(agg_path)
  console.show_status(f'Image saved to `{agg_path}`.')

  i += 1

console.show_status(f'Aggregation task done.')

from ew_viewer import EWViewer
from p3_sleep_graph import plot_sleep_graph
from pictor.objects.signals.signal_group import SignalGroup
from matplotlib import gridspec
from roma import console, finder, io
from tframe.utils.file_tools.io_utils import load

import matplotlib.pyplot as plt
import os
import pandas as pd
import warnings



# Ignore all warnings
warnings.filterwarnings("ignore")

# Configuration
FP_PATH = r'../../features/0211-N153-ch(all)-F(20)-A(128).fp'
EXPORT_PATH = r'../../features/sub-data-01'
AGG_PATH = r'../../features/sub-data-01/report'
XLSX_PATH = r'../../../data/sleep-edf-database-expanded-1.0.0/SC-subjects.xls'
SG_DIR = r'../../../data/sleep-edf-database-expanded-1.0.0/sleep-cassette'

OVERWRITE = 1

# Get sg list and channel list
results = load(FP_PATH)
meta = results['meta']

ew = EWViewer(walker_results=results)

sg_label_list, channel_list, _ = ew.data['meta']

# Load excel
df = pd.read_excel(XLSX_PATH)
PIDs = list(set(df['subject'].tolist()))

N, i = len(PIDs), 0
for pid in PIDs:
  console.show_status(f'Aggregating PID-{pid} ...')
  console.print_progress(i, N)

  agg_path = os.path.join(AGG_PATH, f'{int(pid):03d}.png')
  if os.path.exists(agg_path) and not OVERWRITE:
    N -= 1
    continue

  # Find images
  file_paths = finder.walk(EXPORT_PATH, pattern=f'*SC4{pid:02d}*')
  images = [plt.imread(fp) for fp in file_paths]

  fig = plt.figure(figsize=(20, 10))
  gs = gridspec.GridSpec(2, 3, width_ratios=[2, 2, 4])
  for j, im in enumerate(images):
    gi = j if j < 2 else j + 1
    ax: plt.Axes = plt.subplot(gs[gi])
    ax.set_axis_off()
    ax.imshow(im)

  # Plot sleep graph
  sg_file_list = finder.walk(SG_DIR, pattern=f'*SC*4{pid:02d}??(trim1800;128).sg')
  # assert len(sg_file_list) == 2
  for j, fn in enumerate(sg_file_list):
    sg: SignalGroup = io.load_file(fn, verbose=True)
    ax = plt.subplot(gs[5 if j else 2])
    plot_sleep_graph(ax, sg, line_width=40)

  # Set suptitle based on excel
  indices = df[df['subject'] == int(pid)].index.tolist()
  assert len(indices) >= 1
  row = df.iloc[indices[0]]
  title = f'PID: {pid} | Age: {row["age"]:.0f}'
  title += f' | Gender: {"Male" if row["sex (F=1)"] == 2 else "Female"}'
  plt.suptitle(title)

  # Save image
  plt.tight_layout()
  plt.savefig(agg_path)
  console.show_status(f'Image saved to `{agg_path}`.')

  i += 1

console.show_status(f'Aggregation task done.')

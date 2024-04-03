from pictor.objects.signals.signal_group import SignalGroup
from roma import console, finder, io
from p3_sleep_graph import plot_sleep_graph

import matplotlib.pyplot as plt
import os
import warnings



# Configuration
AGG_PATH = r'../../features/sub-data-02/agg'
REPORT_PATH = r'../../features/sub-data-02/report'
SG_DIR = r'../../../data/rrsh-osa'
OVERWRITE = 1

# Ignore all warnings
warnings.filterwarnings("ignore")

# Find all agg files
agg_file_list = finder.walk(AGG_PATH, pattern='*.png')

N, i = len(agg_file_list), 0
for fn in agg_file_list:
  # Parse PID
  pid = int(fn.split('/')[-1].split('.')[0])

  console.show_status(f'Generating report for PID-{pid} ...')
  console.print_progress(i, N)
  report_path = os.path.join(REPORT_PATH, f'{pid}.png')

  if os.path.exists(report_path) and not OVERWRITE:
    N -= 1
    continue

  # Load image
  agg_im = plt.imread(fn)
  # Find corresponding sg
  sg_file_list = finder.walk(SG_DIR, pattern=f'{pid}(trim;simple;100).sg')
  assert len(sg_file_list) == 1
  sg: SignalGroup = io.load_file(sg_file_list[0], verbose=True)

  # Generate report
  fig: plt.Figure = plt.figure(figsize=(22, 12))
  ax1, ax2 = fig.subplots(2, 1, gridspec_kw={'height_ratios': [10, 1.5]})

  # Show agg image
  ax1.imshow(agg_im)
  ax1.axis('off')

  # Show sleep graph
  plot_sleep_graph(ax2, sg, line_width=20)

  plt.tight_layout()

  # Save image
  plt.savefig(report_path)
  console.show_status(f'Image saved to `{report_path}`.')


console.show_status(f'Report exported.')

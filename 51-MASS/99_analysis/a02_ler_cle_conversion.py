from pictor.objects.signals.eeg import EEG
from pictor.objects.signals.signal_group import SignalGroup
from pictor import Pictor
from roma import finder
from roma import io

import numpy as np
import os



# -----------------------------------------------------------------------------
# (1) Configuration
# -----------------------------------------------------------------------------
MASS_ROOT = r'D:\data\01-MASS'
SSID4VIS = 1

PATTERN = f'0{SSID4VIS}-00**.sg'
PIDS = ['01-0001', '01-0010', '01-0002', '01-0003']
PIDS = ['01-0005', '01-0006', '01-0007', '01-0008']
# -----------------------------------------------------------------------------
# (2) Read .sg files
# -----------------------------------------------------------------------------
sg_file_list = finder.walk(MASS_ROOT, pattern=PATTERN)
sg_file_list = [path for path in sg_file_list
                if any(pid in path for pid in PIDS)]

signal_groups = []
for path in sg_file_list:
  sg: SignalGroup = io.load_file(path, verbose=True)
  signal_groups.append(sg)

N = len(signal_groups)
# -----------------------------------------------------------------------------
# (3) Compare the histograms of LER and CLE
# -----------------------------------------------------------------------------
import matplotlib.pyplot as plt

channel_prefix = ['F3', 'C3', 'O1'][0]

def iqr_norm(data):
  q75, q25 = np.percentile(data, [75, 25])
  iqr = q75 - q25
  return (data - q25) / iqr

# Get data
plotter_labels = ['*',
                  'LER',
                  'CLE',
                  ]
objects = [[] for _ in plotter_labels]
for sg in signal_groups:
  sg = EEG.extract_eeg_channels_from_sg(sg)

  for i, pl in enumerate(plotter_labels):
    if pl == '*':
      chn_cle = f'EEG {channel_prefix}-CLE'
      chn_ler = f'EEG {channel_prefix}-LER'
      chn = chn_cle if chn_cle in sg.channel_signal_dict else chn_ler
    else:
      chn = f'EEG {channel_prefix}-{pl}'

    lb = f'{sg.label} {chn}'
    data = sg[chn]
    objects[i].append((lb, data * 1e6))

def plotter(x: list, fig: plt.Figure):
  # Plot histogram
  assert len(x) == N

  for i in range(N):
    ax = fig.add_subplot(N, 1, i+1)
    lb, data = x[i]
    ax.hist(data, bins=100, alpha=0.5, label=f'{lb}', log=True)
    # ax.set_xlabel('muV')

    ax.set_xlim(-1000, 1000)

    ax.legend()

p = Pictor(figure_size=(10, 5))
p.objects = objects
p.add_plotter(plotter)
p.show()



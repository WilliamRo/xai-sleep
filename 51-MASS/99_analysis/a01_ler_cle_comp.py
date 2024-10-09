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

IQR_NORM = 0
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

channel_prefix = ['F3', 'C3', 'O1']

def iqr_norm(data):
  q75, q25 = np.percentile(data, [75, 25])
  iqr = q75 - q25
  return (data - q25) / iqr

# Get data
objects = [[] for cp in channel_prefix]
for sg in signal_groups:
  for i, cp in enumerate(channel_prefix):
    chn_cle = f'EEG {cp}-CLE'
    chn_ler = f'EEG {cp}-LER'

    chn = chn_cle if chn_cle in sg.channel_signal_dict else chn_ler

    lb = f'{sg.label} {chn}'
    data = sg[chn]
    if IQR_NORM:
      data = iqr_norm(data)
    else:
      data *= 1e6
    objects[i].append((lb, data))

def plotter(x: list, fig: plt.Figure):
  # Plot histogram
  assert len(x) == N

  for i in range(N):
    ax = fig.add_subplot(N, 1, i+1)
    lb, data = x[i]
    ax.hist(data, bins=100, alpha=0.5, label=f'{lb}', log=True)
    # ax.set_xlabel('muV')

    if IQR_NORM: ax.set_xlim(-50, 50)
    else: ax.set_xlim(-1000, 1000)

    ax.legend()

p = Pictor(figure_size=(10, 5))
p.objects = objects
p.add_plotter(plotter)
p.show()



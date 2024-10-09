from docutils.nodes import label

from freud import MASS
from freud.gui.freud_gui import Freud
from pictor.objects.signals.signal_group import SignalGroup
from pictor.objects.signals.eeg import EEG
from roma import finder
from roma import io

import os



# -----------------------------------------------------------------------------
# (1) Configuration
# -----------------------------------------------------------------------------
MASS_ROOT = r'D:\data\01-MASS'
SSID4VIS = 1

# LER
PATTERN = f'0{SSID4VIS}-000[78]*.sg'
CHANNEL_PREFIX = ['F3', 'C3', 'O1'][0]
# -----------------------------------------------------------------------------
# (2) Visualization
# -----------------------------------------------------------------------------
sg_file_list = finder.walk(MASS_ROOT, pattern=PATTERN)

signal_groups = []
for path in sg_file_list:
  sg: SignalGroup = io.load_file(path, verbose=True)

  channels = [
    'EEG F3-CLE',
    'EEG Cz-CLE',
  ]

  sg = sg.extract_channels(channels)
  sg.__class__ = EEG

  key = 'EEG F3-LER'
  sg.digital_signals[0].add_channel(sg[key], key)
  sg = SignalGroup(sg.digital_signals[0], label=sg.label)
  signal_groups.append(sg)

Freud.visualize_signal_groups(signal_groups, 'MASS',
                              default_win_duration=10)

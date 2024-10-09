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
SSID4VIS = 4

N = 5
PATTERN = f'0{SSID4VIS}-00*.sg'

# TODO
# PATTERN = f'0{SSID4VIS}-000[5678]*.sg'
# -----------------------------------------------------------------------------
# (2) Visualization
# -----------------------------------------------------------------------------
sg_file_list = finder.walk(MASS_ROOT, pattern=PATTERN)[:N]

signal_groups = []
for path in sg_file_list:
  sg: SignalGroup = io.load_file(path, verbose=True)

  sg = EEG.extract_eeg_channels_from_sg(sg)

  signal_groups.append(sg)

Freud.visualize_signal_groups(signal_groups, 'MASS',
                              default_win_duration=9999999)

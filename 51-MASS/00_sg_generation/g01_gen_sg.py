from freud import MASS
from freud.gui.freud_gui import Freud
from pictor.objects.signals.signal_group import SignalGroup
from roma import finder
from roma import io

import os



# -----------------------------------------------------------------------------
# (1) Configuration
# -----------------------------------------------------------------------------
MASS_ROOT = r'D:\data\01-MASS'
SSID4VIS = 1

PREPROCESS = ''
# -----------------------------------------------------------------------------
# (2) Generate .sg files subset by subset
# -----------------------------------------------------------------------------
for ss_id in range(1, 6):
  MASS.load_as_signal_groups(data_dir=MASS_ROOT, ssid=ss_id, max_sfreq=128,
                             just_conversion=True)


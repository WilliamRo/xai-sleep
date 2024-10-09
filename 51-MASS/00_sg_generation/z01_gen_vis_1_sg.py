from freud import MASS
from freud.gui.freud_gui import Freud



# -----------------------------------------------------------------------------
# (1) Configuration
# -----------------------------------------------------------------------------
MASS_ROOT = r'D:\data\01-MASS'
PID = '01-0035'

configs = {
  # 'max_sfreq': 128,
}
# -----------------------------------------------------------------------------
# (2) Visualization
# -----------------------------------------------------------------------------
sg = MASS.load_sg_from_raw_files(MASS_ROOT, PID, **configs)

Freud.visualize_signal_groups([sg], 'MASS',
                              default_win_duration=9999999)

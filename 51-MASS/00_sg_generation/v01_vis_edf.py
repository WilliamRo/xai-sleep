from freud.gui.freud_gui import Freud

import os



# -----------------------------------------------------------------------------
# (1) Configuration
# -----------------------------------------------------------------------------
MASS_ROOT = r'F:\01-XAI-SLEEP\01-MASS'

edf_path = os.path.join(MASS_ROOT, 'mass1/01-01-0002 PSG.edf')

# -----------------------------------------------------------------------------
# (2) Visualization
# -----------------------------------------------------------------------------
freud = Freud()
freud.open(edf_path=edf_path)

freud.show()

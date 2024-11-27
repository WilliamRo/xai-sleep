from pictor.xomics.omix import Omix



# -----------------------------------------------------------------------------
# (1) Load omices
# -----------------------------------------------------------------------------
TARGET = [
  'AHI',
  'age'
][0]

# (2.1) Micro
from a01_OSA_omix_micro import omix as micro_omix
from a01_OSA_omix_micro import TARGET as TARGET_1

# (2.2) Macro
from a02_OSA_omix_macro import omix as macro_omix
from a02_OSA_omix_macro import TARGET as TARGET_2

assert TARGET == TARGET_1 == TARGET_2
# -----------------------------------------------------------------------------
# (3) Visualization
# -----------------------------------------------------------------------------
omix = micro_omix * macro_omix

omix.show_in_explorer()

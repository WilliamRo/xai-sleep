from hypnomics.freud.freud import Freud



# -----------------------------------------------------------------------------
# (1) Configuration
# -----------------------------------------------------------------------------
WORK_DIR = r'../data/sleepedfx_sc'

SG_DIR = r'../../data/sleep-edf-database-expanded-1.0.0/sleep-cassette'
SG_PATTERN = f'*(trim1800;128).sg'

OVERWRITE = 1
# -----------------------------------------------------------------------------
# (2) Cloud generation
# -----------------------------------------------------------------------------
freud = Freud(WORK_DIR)

freud.generate_macro_features(SG_DIR, pattern=SG_PATTERN, config='alpha',
                              overwrite=OVERWRITE)

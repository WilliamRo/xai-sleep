from hypnomics.freud.freud import Freud



# -----------------------------------------------------------------------------
# (1) Configuration
# -----------------------------------------------------------------------------
WORK_DIR = r'../data'

SG_DIR = r'../../data/rrsh-osa'
SG_PATTERN = f'*(trim;simple;100).sg'

OVERWRITE = 0
# -----------------------------------------------------------------------------
# (2) Cloud generation
# -----------------------------------------------------------------------------
freud = Freud(WORK_DIR)

freud.generate_macro_features(SG_DIR, pattern=SG_PATTERN, config='alpha',
                              overwrite=OVERWRITE)

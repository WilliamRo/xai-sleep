from hypnomics.freud.freud import Freud
from hf.extractors import get_extractor_dict



# -----------------------------------------------------------------------------
# (1) Configuration
# -----------------------------------------------------------------------------
# The distribution of CLE-referenced EEG data is wider than that of
#  LER-referenced EEG data. Thus, directly extracting cloud can be not
#  appropriate.
WORK_DIR = r'../data/mass_alpha'

SG_DIR = r'D:\data\01-MASS'
SG_PATTERN = f'0?-00??(raw).sg'

CHANNELS = [
  'EEG F3-CLE',
  'EEG F4-CLE',
  'EEG C3-CLE',
  'EEG C4-CLE',
  'EEG O1-CLE',
  'EEG O2-CLE',
  
  'EEG F3-LER',
  'EEG F4-LER',
  'EEG C3-LER',
  'EEG C4-LER',
  'EEG O1-LER',
  'EEG O2-LER',
]

# AASM page size is 30 seconds, while R&K page size is 20 seconds
# Thus, the time resolution should be <10 seconds
TIME_RESOLUTIONS = [
  # 2,
  # 5,
  10,
  # 30,
]
EXTRACTOR_KEYS = [
  'AMP-1',
  'FREQ-20',
  'GFREQ-35',
  # 'P-TOTAL',
  'RP-DELTA',
  'RP-THETA',
  'RP-ALPHA',
  'RP-BETA',
]

OVERWRITE = 0
# -----------------------------------------------------------------------------
# (2) Cloud generation
# -----------------------------------------------------------------------------
freud = Freud(WORK_DIR)

fs = freud.get_sampling_frequency(SG_DIR, SG_PATTERN, CHANNELS)
assert fs == 128

freud.generate_clouds(SG_DIR, pattern=SG_PATTERN, channels=CHANNELS,
                      time_resolutions=TIME_RESOLUTIONS, overwrite=OVERWRITE,
                      extractor_dict=get_extractor_dict(EXTRACTOR_KEYS, fs=fs),
                      channel_should_exist=0)

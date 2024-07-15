from hypnomics.freud.freud import Freud
from hf.extractors import get_extractor_dict



# -----------------------------------------------------------------------------
# (1) Configuration
# -----------------------------------------------------------------------------
# Specify working directory, generated clouds will be put inside
WORK_DIR = r'../data/sleepedfx_sc'

# Specify the directory containing SignalGroups for cloud extraction
SG_DIR = r'../../../data/sleep-edf-database-expanded-1.0.0/sleep-cassette'
SG_PATTERN = f'*(raw).sg'

# Specify EEG channels
CHANNELS = [
  'EEG Fpz-Cz',
  'EEG Pz-Oz'
]

# Specify time resolution, should be factors of 30
TIME_RESOLUTIONS = [
  10,
  30,
]

# Specify probe keys
EXTRACTOR_KEYS = [
  'AMP-1',
  'FREQ-20',
]

# -----------------------------------------------------------------------------
# (2) Generate clouds using Freud
# -----------------------------------------------------------------------------
freud = Freud(WORK_DIR)

fs = freud.get_sampling_frequency(SG_DIR, SG_PATTERN, CHANNELS)
freud.generate_clouds(SG_DIR,
                      pattern=SG_PATTERN,
                      channels=CHANNELS,
                      time_resolutions=TIME_RESOLUTIONS,
                      extractor_dict=get_extractor_dict(EXTRACTOR_KEYS, fs=fs),
                      max_n_sg=4)

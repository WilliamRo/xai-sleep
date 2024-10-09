from mne.io.edf.edf import RawEDF
from fnmatch import fnmatch

import mne



edf_path = r"D:\data\01-MASS\mass1\01-01-0001 PSG.edf"

raw: RawEDF = mne.io.read_raw_edf(edf_path, preload=False)
channel_names = [chn for chn in raw.ch_names if fnmatch(chn, 'EEG*')]

raw: RawEDF = mne.io.read_raw_edf(edf_path, include=channel_names,
                                  preload=True)
import matplotlib.pyplot as plt
raw.plot()
plt.show()

# raw.set_eeg_reference(ref_channels=['A1', 'A2'])

print(raw.info)
from freud.data_io.mne_based import read_digital_signals_mne,read_annotations_mne
import mne
edf_path = r'D:\eason\refer\xai-sleep\data\sleepedfx\SC4001E0-PSG.edf'
edf_path = r'D:\eason\refer\xai-sleep\data\sleepedfx\SC4001EC-Hypnogram.edf'
# raw = mne.io.read_raw_edf(edf_path)
# print(raw.info)
# raw.plot()
# read_digital_signals_mne(edf_path)
read_annotations_mne(edf_path)
from sklearn import preprocessing
import numpy as np

x = np.array([[3., -1., 2., 613.],
              [2., 0., 0., 232],
              [0., 1., -1., 113],
              [1., 2., -3., 489]])
x = x[0, :]
max_abs_scaler = preprocessing.MaxAbsScaler()
x_train_maxsbs = max_abs_scaler.fit_transform(x)

mean = np.mean(x)
std = np.std(x)
print(mean, std)
pass
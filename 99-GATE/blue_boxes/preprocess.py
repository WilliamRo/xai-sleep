from gate_core import th
from slp_datasets.sleepedfx import SleepEDFx
import numpy as np
import matplotlib.pyplot as plt

def cal_iqr(arr):
    qr1 = np.quantile(arr, 0.25)
    qr2 = np.quantile(arr, 0.75)
    iqr = qr2 - qr1
    arr = arr / iqr
    return arr

def sub_meidan(arr):
    median = np.median(arr)
    arr = arr - median
    return arr

def normalize(data):
    arr_mean = np.mean(data)
    arr_std = np.std(data)
    preprocess_data = (data - arr_mean) / arr_std
    return preprocess_data

th.add_noise = False
th.data_config = 'sleepedf:20:0,2,4'
th.ratio = 0
th.overwrite = True
th.show_in_monitor = True
data_name, data_num, channel_select = th.data_config.split(':')
data_set = SleepEDFx.load_as_sleep_set(th.data_dir,
                                       data_name,
                                       data_num,
                                       suffix='-alpha')
sg_groups = data_set.signal_groups
sg = sg_groups[0]
EEG = sg['EEG Fpz-Cz']
EOG = sg['EOG horizontal']
EMG = sg['EMG submental']
x1 = np.arange(EEG.shape[0])
x2 = np.arange(EMG.shape[0])

normalize(EEG)

EEG1 = sub_meidan(EEG)
EOG1 = sub_meidan(EOG)
EMG1 = sub_meidan(EMG)

EEG2 = cal_iqr(EEG1)
EOG2 = cal_iqr(EOG1)
EMG2 = cal_iqr(EMG1)

# plot
fig, ax = plt.subplots(3, 3)
ax[0, 0].plot(x1, EEG)
ax[0, 0].set_title('EEG(original)')
ax[0, 1].plot(x1, EOG)
ax[0, 1].set_title('EOG(original)')
ax[0, 2].plot(x2, EMG)
ax[0, 2].set_title('EMG(original)')
ax[1, 0].plot(x1, EEG1)
ax[1, 0].set_title('EEG(median=0)')
ax[1, 1].plot(x1, EOG1)
ax[1, 1].set_title('EOG(median=0)')
ax[1, 2].plot(x2, EMG1)
ax[1, 2].set_title('EMG(median=0)')
ax[2, 0].plot(x1, EEG2)
ax[2, 0].set_title('EEG(median=0, IQR=1)')
ax[2, 1].plot(x1, EOG2)
ax[2, 1].set_title('EOG(median=0, IQR=1)')
ax[2, 2].plot(x2, EMG2)
ax[2, 2].set_title('EMG(median=0, IQR=1)')
plt.show()
# data_set.configure(channel_select=channel_select)
# data_set.report()
# data_set.show()

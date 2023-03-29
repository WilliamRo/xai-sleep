from gate_core import th
from xsleep.slp_set import SleepSet
from slp_datasets.sleepedfx import SleepEDFx
import numpy as np
import matplotlib.pyplot as plt
import os


th.add_noise = False
th.data_config = 'sleepedfx:20:0'
th.ratio = 0
th.overwrite = True
th.show_in_monitor = True
data_name, data_num, channel_select = th.data_config.split(':')
data_dir = os.path.join(th.data_dir, 'sleepedfx')
sgs1 = SleepEDFx.load_as_signal_groups(data_dir,
                                       first_k=20,)
sgs2 = SleepEDFx.load_as_signal_groups_peiyan(data_dir,
                                              first_k=20,)

data1 = sgs1[10].digital_signals[0].data[:7818000, 0]
data1 = SleepSet.normalize(data1)
data2 = sgs2[10].digital_signals[0].data[:7818000, 0]
data2 = SleepSet.normalize(data2)
path = r'/data/sleepedfx\Sleep_100hz_Novel_CNN_eog_denoise.npy'
data3 = np.load(path)
label1 = sgs1[0].annotations['stage Ground-Truth'].annotations
label2 = sgs2[0].annotations['stage Ground-Truth'].annotations

x = np.arange(data1.shape[0])
x_a = np.arange(len(label1))
fig, ax = plt.subplots(2, 1)
ax[0].plot(x, data1, 'r', x, data3[0], 'b', alpha=0.5)
ax[0].set_title('data')
ax[1].plot(x, data3[0])
ax[1].set_title('data2')

plt.show()


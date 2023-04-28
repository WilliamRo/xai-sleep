from gate_core import th
from xsleep.slp_set import SleepSet
from slp_datasets.sleepedfx import SleepEDFx
import numpy as np
import matplotlib.pyplot as plt
import os


th.add_noise = False
th.data_config = 'sleepedfx:1:0'
th.ratio = 0
th.overwrite = True
th.show_in_monitor = True
data_name, data_num, channel_select = th.data_config.split(':')
data_dir = os.path.join(th.data_dir, 'sleepedfx')
sgs1 = SleepEDFx.load_as_signal_groups(data_dir,
                                       first_k=1,)

fpz_cz = sgs1[0].digital_signals[0].data[:, 1]
resp = sgs1[0].digital_signals[1].data[:30, 1]
max_val = max(fpz_cz)
min_val = min(fpz_cz)
fpz_cz_nomalize = (fpz_cz - min_val) / (max_val - min_val)
resp_nomalize = (resp - min_val) / (max_val - min_val)

x = np.arange(fpz_cz.shape[0])
fig, ax = plt.subplots(2, 1)
ax[0].plot(x[0:100], fpz_cz[0:100], 'r')
ax[0].set_title('fpz_cz')
ax[1].plot(x[0:100], fpz_cz_nomalize[0:100])
ax[1].set_title('fpz_cz_normalize')

plt.show()


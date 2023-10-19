from freud.gui.freud_gui import Freud
from freud.talos_utils.sleep_sets.sleepeason import SleepEason
from fnmatch import fnmatch
from pictor.objects import SignalGroup
from roma import finder
from roma import io
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from sklearn import preprocessing


# Set directories
data_dir = r'../../data/'
data_dir += 'sleepedfx'

prefix = ['', 'sleepedfx', 'ucddb', 'rrshv2'][1]
# pattern = f'{prefix}*.sg'
pattern = f'*SC4001*.sg'
# pattern = f'*2(max_sf*'
pattern = f'*SC**raw*'

# Select .sg files
# T = 42
sg_file_list = finder.walk(data_dir, pattern=pattern)

signal_groups = []

max, min, res = [], [], []
mean, std = [], []
all_data = []
for path in sg_file_list:
  sg = io.load_file(path, verbose=True)
  data = sg.digital_signals[0].data * 1e6
  # data = np.maximum(data, -data)
  all_data.append(data)
  # max.append(np.max(data))
  # min.append(np.min(data))
  # res.append(np.max(data) - np.min(data))
  # mean.append(np.mean(data))
  # std.append(np.std(data))
  # signal_groups.append(sg)


result = np.concatenate(all_data, axis=0)

# data = [signal_groups[T].digital_signals[0].data[:, i] for i in range(3)]
fig, ax = plt.subplots()
n, bins_num, pat = ax.hist(result, bins=10, label=['EEG', 'EEG2', 'EOG'])
# plt.hist(result, bins=10, edgecolor="r", alpha=0.5, label=['EEG', 'EEG2', 'EOG'])
plt.legend()
plt.title('Amplotude distribution')
plt.xlabel("uV")
plt.ylabel("num")
plt.show()

# fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, sharex=True)  # 2行1列的图
# fig, (ax4, ax5) = plt.subplots(2, 1, sharex=True)  # 2行1列的图
x1 = range(len(max))
x2 = range(len(min))

# ax1.plot(x1, max)
# ax1.set_title(' max')
#
# ax2.plot(x2, min)
# ax2.set_title(' min')

# ax3.plot(x2, res)
# ax3.set_title(' res')
#
# ax4.plot(x2, mean)
# ax4.set_title('mean')
#
# ax5.plot(x2, std)
# ax5.set_title('std')
#
# # ax3.set_xlabel('sg id')
# plt.tight_layout()
# plt.show()
#

pass
# filter
# signal_groups[0].digital_signals[0].data = preprocessing(signal_groups[0].digital_signals[0].data)
# t = 1 # 1秒，1000赫兹刻度
# T = 20000
# origin_data = signal_groups[0].digital_signals[0].data[T:T+t*128, 3]
#
# time = np.linspace(0, t, 128 * t)
#
# sos = signal.butter(2, [0.3, 35], btype='bandpass',
#                     analog=False, output='sos',
#                     fs=1000)
# filted_data = signal.sosfilt(sos, origin_data)
#
#
# fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)  # 2行1列的图
# ax1.plot(time, origin_data)
# ax1.set_title('origin data')
#
# ax2.plot(time, filted_data)
# ax2.set_title('After 0.3-35 Hz band-pass filter')
# ax2.set_xlabel('Time [seconds]')
# plt.tight_layout()
# plt.show()
# pass



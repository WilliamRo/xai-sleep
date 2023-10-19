import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
mu = [1.5, 2, 1.5, 1, 1.5]
dis = [0.38, 0.11, 0.35, 0.06, 0.13]
for i in range(5):
  score = math.log(0.2 * mu[i] / dis[i])
  # if
  # print(score)
precision = 0.3910
recall = 0.5635

F1 = 2 * (precision * recall) / (precision + recall)
print(F1)
F1 = 2 * (precision * recall) / max((precision + recall), 1)
print(F1)
from sklearn import preprocessing
x = np.array([[3., -1., 2., 613.],
[2., 0., 0., 232],
[0., 1., -1., 113],
[1., 2., -3., 489]])


# plt.hist(x, bins=10, edgecolor="r", alpha=0.5, label=['EEG', 'EEG2', 'EOG', '4'])
# plt.show()

x_value = [np.random.randint(140, 180, i) for i in [100, 200, 300]]

plt.hist(x_value, bins=10, edgecolor="r", histtype="bar", alpha=0.5,
         label=["A公司", "B公司", "C公司"])
plt.legend()
plt.title('asd')
plt.xlabel("1e-6")
plt.ylabel("num")
# plt.xlim(0, 100)
# plt.show()



import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# b, a = signal.butter(4, 100, 'low', analog=True)  # 4阶低通临界频率为100Hz
# w, h = signal.freqs(b, a)  # h为频率响应,w为频率
# plt.figure(1)
# plt.semilogx(w, 20 * np.log10(abs(h)))  # 用于绘制折线图，两个函数的 x 轴、y 轴分别是指数型的，并且转化为分贝
# plt.title('Butterworth filter frequency response')
# plt.xlabel('Frequency [radians / second]')
# plt.ylabel('Amplitude [dB]')
# plt.margins(0, 0.1)  # 去除画布四周白边
# plt.grid(which='both', axis='both')  # 生成网格，matplotlin.pyplot.grid(b, which, axis, color, linestyle, linewidth， **kwargs)， which : 取值为'major', 'minor'， 'both'
# plt.axvline(100, color='green')  # 绘制竖线
# plt.show()
# t = np.linspace(0, 1, 1000, False)  # 1秒，1000赫兹刻度
# sig = np.sin(2*np.pi*10*t) + np.sin(2*np.pi*20*t)  # 合成信号
# fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)  # 2行1列的图
# ax1.plot(t, sig)
# ax1.set_title('10 Hz and 20 Hz sinusoids')
# ax1.axis([0, 1, -2, 2])  # 坐标范围

# t = np.linspace(0, 1, 1000, False)  # 1秒，1000赫兹刻度
# sig = np.sin(2*np.pi*10*t) + np.sin(2*np.pi*20*t)  # 合成信号
# fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)  # 2行1列的图
# ax1.plot(t, sig)
# ax1.set_title('10 Hz and 20 Hz sinusoids')
# ax1.axis([0, 1, -2, 2])  # 坐标范围
# #sos = signal.butter(10, 15, 'hp', fs=1000, output='sos')  #10阶，15赫兹
# # sos = signal.butter(10, 10, btype='low', analog=False, output='sos', fs=1000)
# sos = signal.butter(10, [1, 40], btype='bandpass', analog=False, output='sos', fs=1000)
# filtered = signal.sosfilt(sos, sig)  # 滤波
# ax2.plot(t, filtered)
# ax2.set_title('After 15 Hz high-pass filter')
# ax2.axis([0, 1, -2, 2])
# ax2.set_xlabel('Time [seconds]')
# plt.tight_layout()
# plt.show()

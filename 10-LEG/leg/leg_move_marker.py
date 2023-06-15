from pictor.objects.signals.signal_group import SignalGroup, Annotation
from scipy import signal
from roma import console
from scipy.signal import hilbert, chirp
import numpy as np
import matplotlib.pyplot as plt



# 画包络超级慢，要32s左右，趋势线只要9s
def mark_single_channel_alpha1(y, fs):
  """Returns a list of intervals (indices): [(i1, i2), (i3, i4), ...]"""

  # initialize
  intervals = []
  y = np.abs(y * 10 ** (6))
  base = 0 #设置静息状态值

  # 包络线
  analytic_signal = hilbert(y)
  amplitude_envelope = np.abs(analytic_signal)

  fs = int(fs)
  i = 0

  L = len(y)
  while i < L:
    console.print_progress(i, L)

    #看是否更新静息状态
    sig = y[i]

    n = 0
    h = min(i + fs * 0.5, len(y))
    if max(amplitude_envelope[i:h]) - min(amplitude_envelope[i:h]) < 1:
      n += 1
      i += fs * 0.5
      continue
    elif n > 15:
      base = np.mean(i - fs * 0.5, i)

    #找大于静息状态+8μV的信号
    if sig>(base+8):
        duration = 0
        for j in range(i,i+10*fs,fs//4):
          data = np.mean(y[j:(j+fs//4)])
          if np.mean(y[j:(j+fs//4)])<(base+2):
              break
          duration += fs//4
        if i-intervals[-1][1] < fs: intervals[-1][1] += duration
        elif duration>=fs//2: intervals.append((i, i+duration))
        i = i + duration
    i += 1

  #4. 考虑睡眠分期，只选择睡眠期间的腿动事件
  return intervals


def mark_single_channel_alpha(y, fs):
  """Returns a list of intervals (indices): [(i1, i2), (i3, i4), ...]"""

  # initialize
  intervals = []
  y = np.abs(y * 10 ** (6))
  base = 0 #设置静息状态值

  #趋势线
  b, a = signal.butter(4, 0.05)
  trend = signal.filtfilt(b, a, y)

  fs = int(fs)
  i = 0

  L = len(y)
  while i < L:
    console.print_progress(i, L)

    #看是否更新静息状态
    sig = y[i]
    n = 0
    h = min(i + fs * 1, len(y))
    if max(trend[i:h]) - min(trend[i:h]) < 1:
      n += 1
      i += fs * 1
      continue
    elif n > 3:
      base = np.mean(y[i - fs * 1: i])

    #找大于静息状态+8μV的信号
    if sig>(base+8):
        duration = 0
        for j in range(i,i+50*fs,fs//4):
          if np.mean(y[j:(j+fs//4)])<(base+2):
              break
          duration += fs//4

        if intervals and i - intervals[-1][1] < fs * 0.3:
          intervals[-1] = (intervals[-1][0], i + duration)
          # intervals[-1][1] = last + duration
        elif fs // 2 <= duration and duration < fs * 10:
          intervals.append((i, i + duration))
        i = i + duration
    i += 1

  #4. 考虑睡眠分期，只选择睡眠期间的腿动事件
  return intervals

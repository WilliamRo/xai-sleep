import numpy as np
from pictor.objects.signals.signal_group import SignalGroup, Annotation
from scipy import signal
import matplotlib.pyplot as plt



def mark_single_channel_alpha(y, fs):
  """Returns a list of intervals (indices): [(i1, i2), (i3, i4), ...]"""
  intervals = []
  y = np.abs(y * 10 ** (6))
  #设置静息状态值
  base = 0

  #趋势线
  b, a = signal.butter(4, 0.05)
  trend = signal.filtfilt(b, a, y)

  fs = int(fs)
  i = 0
  while i<len(y):
    #看是否更新静息状态
    sig = y[i]
    h = min(i + fs * 8, len(y))
    if max(trend[i:h]) - min(trend[i:h]) < 1:
      base = np.mean(y[i:h])
      # TODO:
      # print(base)
      i += fs * 8
      continue

    #找大于静息状态+8μV的信号
    if sig>(base+8):
        duration = 0
        for j in range(i,i+10*fs,fs//4):
          data = np.mean(y[j:(j+fs//4)])
          if np.mean(y[j:(j+fs//4)])<(base+2):
              break
          duration += fs//4
        if duration>=fs//2:
          intervals.append((i, i+duration))
        i = i + duration
    i += 1

  #4. 考虑睡眠分期，只选择睡眠期间的腿动事件
  return intervals

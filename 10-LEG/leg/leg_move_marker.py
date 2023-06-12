import numpy as np
from pictor.objects.signals.signal_group import SignalGroup, Annotation
from scipy import signal
import matplotlib.pyplot as plt



def marker_alpha(sg: SignalGroup,channel_key) -> Annotation:
  anno = {}
  anno['start'] = []
  anno['duration'] = []

  #1. get data
  data = sg.name_tick_data_dict[channel_key]
  y = data[1]

  #2. filter
  #高通滤波
  fs = sg.dominate_signal.sfreq
  f = 10
  b, a = signal.butter(4, f, btype='highpass', fs=fs)
  y = signal.lfilter(b, a, y)

  #去掉50Hz的噪声频率
  f0 = 50
  Q = 10
  b, a = signal.iirnotch(f0, Q, fs=fs)
  y = signal.lfilter(b, a, y)
  # np.save('P:/XAI/y.npy', y)
  y = np.abs(y * 10 ** (6))

  #3. 找到大于8μV的起点，小于2μV的终点
  #设置静息状态值
  base = 0

  #趋势线
  b, a = signal.butter(4, 0.05)
  envelope = signal.filtfilt(b, a, y)

  fs = int(fs)
  i = 0
  while i<len(y):
    #看是否更新静息状态
    sig = y[i]
    h = min(i + fs * 8, len(y))
    uu = envelope[i:h]

    if max(uu) - min(uu) < 1:
      base = np.mean(y[i:h])
      # TODO:
      # print(base)
      i += fs * 8
      continue

    #找大于静息状态+8μV的信号
    if sig>(base+8):
        l = 0
        for j in range(i,i+10*fs,fs//4):
          data = np.mean(y[j:(j+fs//4)])
          if np.mean(y[j:(j+fs//4)])<(base+2):
              break
          l += fs//4
        if l>=fs//2:
          anno['start'].append(i/fs)
          anno['duration'].append(l/fs)
        i = i + l
    i += 1

  #4. 考虑睡眠分期，只选择睡眠期间的腿动事件
  return anno

def marker_beta(sg: SignalGroup,channel_key) -> Annotation:
  anno = {}
  anno['start'] = []
  anno['duration'] = []

  import xml.dom.minidom as xml
  fn = r'../../data/rrsh/111.xml'
  dom = xml.parse(fn)
  root = dom.documentElement
  events = root.getElementsByTagName('ScoredEvent')

  if channel_key == 'Leg/L':
    channel = 'Limb Movement (Left)'
  elif channel_key == 'Leg/R':
    channel = 'Limb Movement (Right)'

  for i, event in enumerate(events):
    label = event.childNodes
    if label[0].childNodes[0].data == channel:
      anno['start'].append(float(label[1].childNodes[0].data))
      anno['duration'].append(float(label[2].childNodes[0].data))

  return anno
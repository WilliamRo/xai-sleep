from pictor.objects.signals.signal_group import SignalGroup, Annotation
from scipy import signal
from leg.leg_monitor import LegMonitor
from roma import Nomear

import matplotlib.pyplot as plt
import numpy as np



class LegObserver(LegMonitor):
  pass



class LegEMG(Nomear):

  def __init__(self, recording, fs):
    self.recording = recording
    self.fs = fs

  @Nomear.property()
  def envelop(self):
    return



if __name__ == '__main__':
  from freud.talos_utils.sleep_sets.rrshv1 import RRSHSCv1
  from tframe import console

  from leg.even import Even

  console.suppress_logging()
  data_dir = r'../../data/rrsh-mini'
  signal_groups = RRSHSCv1.load_as_signal_groups(data_dir)

  signal_segments = []
  sg = signal_groups[0]
  seg_indices = [
    (26795, 26820), (27460, 27490),
  ]
  for t1, t2 in seg_indices:
    signal_segments.append(sg.truncate(t1, t2, return_new_sg=True))

  even = Even(title='Even')
  even.objects = signal_segments
  # even.objects = signal_groups
  even.monitor.set('channels', 'E1-M2,Leg/L,Leg/R', auto_refresh=False)

  even.show()

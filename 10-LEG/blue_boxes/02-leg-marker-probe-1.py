from freud.talos_utils.sleep_sets.rrshv1 import RRSHSCv1
from tframe import console

from leg.even import Even



console.suppress_logging()
data_dir = r'../../data/rrsh'
signal_groups = RRSHSCv1.load_as_signal_groups(data_dir)


even = Even(title='Even')
signal_segments = []

# TODO:
sg = signal_groups[0]
seg_indices = [
  (26430, 26455), (26740, 26800),
]

for t1, t2 in seg_indices:
  signal_segments.append(sg.truncate(t1, t2, return_new_sg=True))

even.objects = signal_segments

even.monitor.set('channels', 'E1-M2,E2-M2,Leg/L,Leg/R', auto_refresh=False)

even.show()

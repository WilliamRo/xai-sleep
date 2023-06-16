from freud.talos_utils.sleep_sets.rrshv1 import RRSHSCv1
from tframe import console

from leg.even import Even



console.suppress_logging()
data_dir = r'../../data/rrsh'
signal_groups = RRSHSCv1.load_as_signal_groups(data_dir)


even = Even(title='Even')
even.objects = signal_groups
channels = ['E1-M2', 'E2-M2', 'Chin 1-Chin 2', 'Nasal Pressure',
            'Leg/L', 'Leg/R', 'PositionSen']
even.monitor.set('channels', ','.join(channels), auto_refresh=False)

even.show()

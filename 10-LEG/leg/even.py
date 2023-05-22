from freud.data_io.mne_based import read_digital_signals_mne
from freud.gui.sleep_monitor import SleepMonitor
from freud.gui.freud_gui import Freud
from pictor import Pictor
from pictor.objects.signals.signal_group import SignalGroup, Annotation
from pictor.plugins import DialogUtilities
from roma import console

from leg.leg_monitor import LegMonitor



class Even(Freud):

  def __init__(self, title='Even', figure_size=(12, 8)):
    super(Freud, self).__init__(title, figure_size=figure_size)

    self.monitor: LegMonitor = self.add_plotter(LegMonitor(self))

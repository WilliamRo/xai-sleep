from cam_oliver.cam_monitor import CAMonitor
from freud.data_io.mne_based import read_digital_signals_mne
from freud.gui.sleep_monitor import SleepMonitor
from freud.gui.freud_gui import Freud
from pictor import Pictor
from pictor.objects.signals.signal_group import SignalGroup, Annotation
from pictor.plugins import DialogUtilities
from roma import console



class Oliver(Freud):

  def __init__(self, title='Oliver', figure_size=(12, 8)):
    super(Freud, self).__init__(title, figure_size=figure_size)

    self.monitor = self.add_plotter(CAMonitor(self))

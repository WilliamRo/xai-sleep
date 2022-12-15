from pictor import Pictor
from freud.gui.sleep_monitor import SleepMonitor



class Freud(Pictor):

  def __init__(self, title='Freud', figure_size=(12, 8)):
    super(Freud, self).__init__(title, figure_size=figure_size)

    self.monitor = self.add_plotter(SleepMonitor())



if __name__ == '__main__':
  freud = Freud()
  freud.show()

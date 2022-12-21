from pictor.plotters import Monitor



class SleepMonitor(Monitor):

  def register_shortcuts(self):
    super(SleepMonitor, self).register_shortcuts()

    self.register_a_shortcut('O', self.pictor.open,
                             description='Open a .edf file')

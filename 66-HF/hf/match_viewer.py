from pictor.plotters import Plotter

import matplotlib.pyplot as plt
import numpy as np



class MatchViewer(Plotter):

  def __init__(self, brick: np.ndarray):
    super().__init__(self.plot, None)
    self.brick = brick


  def plot(self, ax: plt.Axes, x):
    M = self.brick[:, :, x]

    ax.matshow(M, cmap='viridis')


  def analyze(self):
    from pictor import Pictor

    p = Pictor.image_viewer('')
    p.plotters[0].set('cmap', 'RdYlGn')
    p.plotters[0].set('color_bar', True)
    p.plotters[0].set('title', True)
    p.objects = list(range(self.brick.shape[2]))
    p.labels = labels
    p.show()
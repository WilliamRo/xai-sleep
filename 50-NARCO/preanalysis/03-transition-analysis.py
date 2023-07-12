from freud.talos_utils.sleep_sets.rrshv1 import RRSHSCv1
from pictor.plotters.plotter_base import Plotter
from tframe import console

import matplotlib.pyplot as plt
import numpy as np



class TransitionAnalyzer(Plotter):

  def __init__(self, pictor=None):
    # Call parent's constructor
    super().__init__(self.show_scatter, pictor)

  def show_scatter(self, x, ax: plt.Axes):
    title, pcts = x

    xs = np.linspace(-10, 10)
    ys = np.random.random(xs.size)

    ax.scatter(xs, ys)



if __name__ == '__main__':
  # Load data
  def load_stages(path):
    transitions = []
    signal_groups = RRSHSCv1.load_as_signal_groups(path)
    for sg in signal_groups:
      anno = sg.annotations[RRSHSCv1.ANNO_KEY_GT_STAGE]
      stages = anno.annotations
      labels = ['W', '1', '2', '3', 'R']
      brief = [stages[0]]
      for s in stages:
        if s != brief[-1] and s < 5: brief.append(s)
      brief = [labels[i] for i in brief if i < 5]
      transitions.append(brief)
    return transitions

  data_dir = [
    r'../../data/rrsh',
    r'../../data/rrsh-narcolepsy',
  ]
  transitions = []
  for path in data_dir: transitions.extend(load_stages(path))

  # Create objects
  objects = {}
  STAGES = ['W', '1', '2', '3', 'R']

  for s1 in STAGES:
    for s2 in STAGES:
      counts = [0 for t in transitions]
      for pid, stages in enumerate(transitions):
        N = len(stages)
        for i in range(N - 1):
          if True: counts[pid] += 1
        counts[pid] = counts[pid] / (N -1)

      objects[s1+s2] = counts

  # Show
  objects = [(k, v) for k, v in objects.items()]
  TransitionAnalyzer.plot(objects, fig_size=(16, 8))
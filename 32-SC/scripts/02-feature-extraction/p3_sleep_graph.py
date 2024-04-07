from roma import finder
from roma import io
from pictor.objects.signals.signal_group import SignalGroup

import matplotlib.pyplot as plt



def plot_sleep_graph(ax: plt.Axes, sg: SignalGroup, line_width=20):
  # Extract annotations
  anno = sg.annotations['stage Ground-Truth']
  t, stages = anno.curve
  t = t / 3600

  is_valid = lambda s: 0 <= s <= 4
  # Plot sleep graph
  colors = ['forestgreen', 'gold', 'orange', 'royalblue', 'lightcoral']
  for i, c in enumerate(colors):
    ax.plot([t[0], t[-1]], [i, i], color=c, linewidth=line_width, alpha=0.2)
  # for y in range(5): plt.plot([t[0], t[-1]], [y, y], color='#F0F0F0')
  trashcan = None
  for i in range(len(t) - 1):
    t1, t2 = t[i], t[i+1]
    s1, s2 = stages[i], stages[i+1]
    if is_valid(s1) and is_valid(s2):
      if trashcan:
        ax.plot([trashcan[0], t1], [trashcan[1], s1], color='#DDD', alpha=0.8)
        trashcan = None
      ax.plot([t1, t2], [s1, s2], color='black', alpha=0.8)
    elif not trashcan:
      assert is_valid(s1)
      trashcan = (t1, s1)
  # ax.plot(t, stages, color='black', alpha=0.8)

  ax.set_xlim(t[0], t[-1])
  ax.set_xlabel('Time (hour)')

  # ax.set_ylim(-0.5, 4.5)
  ax.set_yticks([0, 1, 2, 3, 4], ['W', 'N1', 'N2', 'N3', 'R'])

  ax.invert_yaxis()



if __name__ == '__main__':
  # Configs
  N = 2

  # Select .sg files
  data_dir = r'../../../data/rrsh-osa'
  pattern = f'*(trim;simple;100).sg'

  sg_file_list = finder.walk(data_dir, pattern=pattern)[:N]
  sg: SignalGroup = io.load_file(sg_file_list[0], verbose=True)

  # Plot
  fig: plt.Figure = plt.figure(figsize=(12, 3))
  ax = fig.subplots()
  plot_sleep_graph(ax, sg)
  plt.tight_layout()
  plt.show()




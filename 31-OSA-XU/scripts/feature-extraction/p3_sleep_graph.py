from roma import finder
from roma import io
from pictor.objects.signals.signal_group import SignalGroup

import matplotlib.pyplot as plt



def plot_sleep_graph(ax: plt.Axes, sg: SignalGroup, line_width=20):
  # Extract annotations
  anno = sg.annotations['stage Ground-Truth']
  t, stages = curve = anno.curve
  t = t / 3600

  # Plot sleep graph
  colors = ['forestgreen', 'gold', 'orange', 'royalblue', 'lightcoral']
  for i, c in enumerate(colors):
    ax.plot([t[0], t[-1]], [i, i], color=c, linewidth=line_width, alpha=0.2)
  # for y in range(5): plt.plot([t[0], t[-1]], [y, y], color='#F0F0F0')
  ax.plot(t, stages, color='black', alpha=0.8)

  ax.set_xlim(t[0], t[-1])
  ax.set_xlabel('Time (hour)')

  ax.set_yticks([0, 1, 2, 3, 4], ['W', 'N1', 'N2', 'N3', 'R'])

  ax.invert_yaxis()



def plot_sleep_graph_clean(ax: plt.Axes, sg: SignalGroup, line_width=20):
  # Extract annotations
  anno = sg.annotations['stage Ground-Truth']
  t, stages = curve = anno.curve
  t = t / 3600

  # Plot sleep graph
  for y in range(5): plt.plot([t[0], t[-1]], [y, y], color='#F0F0F0')
  ax.plot(t, stages, color='black', alpha=0.8)

  ax.set_xlim(t[0], t[-1])
  # ax.set_xlabel('Time (hour)')
  ax.set_xticks([])

  ax.set_yticks([0, 1, 2, 3, 4], ['W', 'N1', 'N2', 'N3', 'R'])

  ax.invert_yaxis()



if __name__ == '__main__':
  # Configs
  N = 10

  # Select .sg files
  data_dir = r'../../../data/rrsh-osa/rrsh_osa_sg'
  pattern = f'*(trim;simple;100).sg'

  sg_file_list = finder.walk(data_dir, pattern=pattern)[:N]
  sg: SignalGroup = io.load_file(sg_file_list[0], verbose=True)

  # Plot
  fig: plt.Figure = plt.figure(figsize=(6, 1))
  ax = fig.subplots()
  plot_sleep_graph(ax, sg)
  # plot_sleep_graph_clean(ax, sg)
  plt.tight_layout()
  plt.show()




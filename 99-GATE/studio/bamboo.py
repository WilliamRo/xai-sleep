import numpy as np

from pictor import Pictor
from tframe.utils.note import Note

import matplotlib.pyplot as plt


sum_path = r'E:\wanglin\project\deep_learning\xai-sleep\99-GATE\01_data_fusion\1212_s99_allchn_noise_unknown.sum'
sum_path1 = r'E:\wanglin\project\deep_learning\xai-sleep\99-GATE\01_data_fusion\1212_s99_allchn_noise_zero.sum'

# load notes
notes = Note.load(sum_path)
# notes1 = Note.load(sum_path1)

# config
ratio = [0.1 * i for i in range(11)]
xlabel = [10 * i for i in range(11)]
# curve1
notes_ratio_all = []
mean_accuracy_all = []
confidence_high_all = []
confidence_low_all = []
# curve2
notes_ratio_all1 = []
mean_accuracy_all1 = []

# classify notes by ratio
for i in ratio:
  notes_per_ratio = [n for n in notes if n.configs['ratio'] == i]
  notes_ratio_all.append(notes_per_ratio)

# for i in ratio:
#   notes_per_ratio1 = [n for n in notes1 if n.configs['ratio'] == i]
#   notes_ratio_all1.append(notes_per_ratio1)

# calculate statistical parameter in per ratio
for notes in notes_ratio_all:
  accuracy = [n.criteria['Test Accuracy'] for n in notes]
  mean = np.mean(accuracy)
  mean_accuracy_all.append(mean)
  # calculate confidence interval
  std = np.std(accuracy)
  confidence_high = mean + 1.96 * std / np.sqrt(len(accuracy))
  confidence_low = mean - 1.96 * std / np.sqrt(len(accuracy))
  confidence_high_all.append(confidence_high)
  confidence_low_all.append(confidence_low)

# for notes in notes_ratio_all1:
#   accuracy = [n.criteria['Test Accuracy'] for n in notes]
#   mean = np.mean(accuracy)
#   mean_accuracy_all1.append(mean)

def plotter(ax: plt.Axes):
  ax.plot(xlabel, mean_accuracy_all, color='red', linewidth=1, alpha=1,
          marker='s', label='data_fusion')
  ax.fill_between(xlabel, confidence_low_all, confidence_high_all, alpha=0.3, color='red')
  # ax.plot(xlabel, mean_accuracy_all1, color='blue', linewidth=2, alpha=1,
  #         marker='s', label='feature_fusion')
  ax.set_xlabel('p(%)', loc='right')
  ax.set_ylabel('accuracy(%)', loc='top')
  ax.set_title('unknown data',fontweight='bold', loc='center')
  ax.legend()

p = Pictor(figure_size=(8, 5))
plotter = p.add_plotter(plotter)
p.show()


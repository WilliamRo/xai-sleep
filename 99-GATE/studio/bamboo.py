import numpy as np

from pictor import Pictor
from tframe.utils.note import Note

import matplotlib.pyplot as plt


# sum_path = r'E:\wanglin\project\deep_learning\xai-sleep\99-GATE\02_feature_fusion\1215_s99_allchn_noise_unknown_beta.sum'
# sum_path2 = r'E:\wanglin\project\deep_learning\xai-sleep\99-GATE\02_feature_fusion\1215_s99_allchn_noise_zero_beta.sum'
sum_path = r'E:\wanglin\project\deep_learning\xai-sleep\99-GATE\01_data_fusion\1212_s99_allchn_noise_unknown.sum'
sum_path2 = r'E:\wanglin\project\deep_learning\xai-sleep\99-GATE\01_data_fusion\1212_s99_allchn_noise_zero.sum'

# load notes
notes = Note.load(sum_path)
notes2 = Note.load(sum_path2)

# config
ratio = [0.1 * i for i in range(11)]
xlabel = [10 * i for i in range(11)]
# curve1
notes_ratio_all = []
mean_accuracy_all = []
confidence_high_all = []
confidence_low_all = []
# curve2
notes_ratio_all2 = []
mean_accuracy_all2 = []
confidence_high_all2 = []
confidence_low_all2 = []

# classify notes by ratio
for i in ratio:
  notes_per_ratio = [n for n in notes if n.configs['ratio'] == i]
  notes_ratio_all.append(notes_per_ratio)

for i in ratio:
  notes_per_ratio2 = [n for n in notes2 if n.configs['ratio'] == i]
  notes_ratio_all2.append(notes_per_ratio2)

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

for notes in notes_ratio_all2:
  accuracy = [n.criteria['Test Accuracy'] for n in notes]
  mean = np.mean(accuracy)
  mean_accuracy_all2.append(mean)
  # calculate confidence interval
  std = np.std(accuracy)
  confidence_high = mean + 1.96 * std / np.sqrt(len(accuracy))
  confidence_low = mean - 1.96 * std / np.sqrt(len(accuracy))
  confidence_high_all2.append(confidence_high)
  confidence_low_all2.append(confidence_low)

def plotter(ax: plt.Axes):
  ax.plot(xlabel, mean_accuracy_all, color='red', linewidth=1, alpha=1,
          marker='s', label='unknown')
  ax.fill_between(xlabel, confidence_low_all, confidence_high_all,
                  alpha=0.3, color='red')
  ax.plot(xlabel, mean_accuracy_all2, color='blue', linewidth=1, alpha=1,
          marker='s', label='zero')
  ax.fill_between(xlabel, confidence_low_all2, confidence_high_all2,
                  alpha=0.3, color='blue')
  ax.plot(xlabel, np.ones(len(xlabel)) * 48.99, color='gray', linewidth=2,
          alpha=0.9, linestyle='dashed', label='EMG')
  ax.plot(xlabel, np.ones(len(xlabel)) * 66.67, color='gray', linewidth=2,
          alpha=0.9, linestyle='-.', label='EOG')
  ax.plot(xlabel, np.ones(len(xlabel)) * 75.36, color='gray', linewidth=2,
          alpha=0.9, linestyle='-', label='EEG')
  ax.set_xlabel('p(%)', loc='right')
  ax.set_ylabel('accuracy(%)', loc='top')
  ax.set_title('data fusion',fontweight='bold', loc='center')
  ax.legend()

p = Pictor(figure_size=(8, 5))
plotter = p.add_plotter(plotter)
p.show()


import numpy as np

from pictor import Pictor
from tframe.utils.note import Note
import pickle5 as p5
import pickle
import matplotlib.pyplot as plt


# sum_path = r'E:\wanglin\project\deep_learning\xai-sleep\00-XSLP\02_convnet\1208_s00_convnet.sum'
sum_path = r'E:\wanglin\project\deep_learning\xai-sleep\00-XSLP\02_feature_fusion\1215_s00_feature_fusion_convnet.sum'

# load notes
try:
  notes = Note.load(sum_path)
except ValueError:
  with open(sum_path, 'rb') as f:
    notes = p5.load(f)
  with open(sum_path, 'wb') as f:
    # convert sum to protocal 4
    pickle.dump(notes, f, pickle.HIGHEST_PROTOCOL)
# config
data_configs = ['sleepedfx:20:0,2,4', 'sleepedfx:20:0,1,2']

notes_all = []
mean_accuracy_all = []
mean_f1_all = []
mean_precision_all = []
mean_recall_all = []

# classify notes by ratio
for i in data_configs:
  notes_per_channel = [n for n in notes if n.configs['data_config'] == i]
  notes_all.append(notes_per_channel)

# calculate mean accuracy in per ratio
for notes in notes_all:
  accuracy = [n.criteria['Test Accuracy'] for n in notes]
  f1 = [n.misc['confusion_m'].macro_F1 for n in notes]
  precision = [n.misc['confusion_m'].macro_precision for n in notes]
  recall = [n.misc['confusion_m'].macro_recall for n in notes]
  mean_accuracy = np.mean(accuracy)
  mean_f1 = np.mean(f1)
  mean_precision = np.mean(precision)
  mean_recall = np.mean(recall)
  mean_accuracy_all.append(mean_accuracy)
  mean_f1_all.append(mean_f1)
  mean_precision_all.append(mean_precision)
  mean_recall_all.append(mean_recall)

print(mean_accuracy_all)
print(mean_f1_all)
print(mean_precision_all)
print(mean_recall_all)

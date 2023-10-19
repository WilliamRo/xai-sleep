import sys, os
import numpy as np
from pictor import Pictor
from tframe.utils.note import Note
from eason_boxes.plot import Plot
import matplotlib.pyplot as plt

sum_path = r'D:\eason\refer\xai-sleep\06-ATTN\06_attn_k_fold\1009_s6_mask.sum'

# load notes
notes = Note.load(sum_path)


# region: classify according to key
# dic = {}
# for i in range(len(notes)):
#   # key = notes[i].configs['data_config'].split(' ')[2]
#   key = notes[i].configs['epoch_pad']
#   value = notes[i].criteria['Test F1']
#   if key in dic:
#     dic[key].append(value)
#   else:
#     dic[key] = [value]
# # endregion: classify according to key
#
# plot1 = Plot(dic, 'pad', 'mf1', 'attn-sleep')
# plot1.show()


# region: classify according to key
dic2 = {}
for i in range(len(notes)):
  key1 = notes[i].configs['data_config'].split(' ')[2]
  key2 = notes[i].configs['epoch_pad']
  key = key1 + ' pad:' + str(key2)
  value = notes[i].criteria['Test F1']
  if key in dic2:
    dic2[key].append(value)
  else:
    dic2[key] = [value]

new_dic = {}
filt_key = 'pad:0'
for key, value in dic2.items():
  if filt_key in key:
    new_key = key[:7]
    new_dic[new_key] = value
# endregion: classify according to key


plot2 = Plot(new_dic, 'fold', 'mf1', filt_key, 'attn-sleep')
# plot1.cal_data_from_dic(dic)
plot2.show()



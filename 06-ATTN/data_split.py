import scipy.io
import pandas as pd
from roma.spqr.finder import walk
from collections import defaultdict
import numpy as np




# data_path
data_path = r'.\data_split_eval.mat'
data_dir = r'..\data\sleepeasonx'
def split_xsleep(path, file_lists):
  mat_data = scipy.io.loadmat(path)
  train_key = mat_data['train_sub']
  val_key = mat_data['eval_sub']
  test_key = mat_data['test_sub']


  k_fold_key = []
  for val, test in zip(val_key, test_key):
    dic = {}
    dic['val'] = val[0].tolist()[0]
    dic['test'] = test[0].tolist()[0]
    k_fold_key.append(dic)

  map_dict = {}
  for sg_list in file_lists:
    # sg_list e.g. '../../sleepedf-SC4001E.sg'
    # patient_id = int('00')
    patient_id = int(sg_list.split("SC4")[-1][:2])

    # map_dict e.g. {0:'SC400'}
    map_dict[patient_id] = sg_list.split('/')[-1].split('-')[1][:5]




  return k_fold_key, map_dict

file_lists = walk(data_dir, pattern='*sg')
fold_sets, map = split_xsleep(data_path, file_lists)

BENCHMARK = {}
keys = list(map.keys())
for i in range(len(fold_sets)):
  label = f'gamma{i+1}'
  BENCHMARK[label] = {}

  test_ids = fold_sets[i]['test']
  val_ids = fold_sets[i]['val']

  BENCHMARK[label]['val'] = [map[keys[id-1]] for id in val_ids]
  BENCHMARK[label]['test'] = [map[keys[id-1]] for id in test_ids]




pass

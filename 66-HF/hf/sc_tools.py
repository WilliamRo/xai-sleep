from collections import OrderedDict
from hypnomics.freud.freud import Freud
from hypnomics.freud.nebula import Nebula
from roma import io

import numpy as np
import os
import pandas as pd



def get_paired_sg_labels(sg_labels: list, excludes=(), return_two_lists=False):
  """e.g., label = `SC4012E`"""
  # (1) Group by subjects
  od = OrderedDict()
  for label in sg_labels:
    pid = label[2:5]
    if pid not in od: od[pid] = []
    od[pid].append(label)

  # Remove single-night subjects
  for k in list(od.keys()):
    if len(od[k]) < 2: od.pop(k)

  if return_two_lists:
    nights_1, nights_2 = [], []
    for labels in od.values():
      if labels[0][2:5] not in excludes:
        nights_1.append(labels[0])
        nights_2.append(labels[1])
    return nights_1, nights_2

  paired_sg_labels = []
  for labels in od.values():
    if labels[0][2:5] not in excludes:
      paired_sg_labels.extend(labels)

  # pids = [sg_label[2:5] for sg_label in sg_labels]
  # for label, pid in zip(sg_labels, pids):
  #   if len([p for p in pids if p == pid]) > 1:
  #     if pid not in excludes: paired_sg_labels.append(label)

  return paired_sg_labels


def get_dual_nebula(nebula: Nebula):
  night_1, buffer_1 = [], []
  night_2, buffer_2 = [], []

  for label in nebula.labels:
    pid = label[2:5]
    if pid not in buffer_1:
      buffer_1.append(pid)
      night_1.append(label)
    else:
      assert pid not in buffer_2
      buffer_2.append(pid)
      night_2.append(label)

  return nebula[night_1], nebula[night_2]


def load_sc_meta(XLSX_PATH, SG_LABELS):
  meta_dict = {}
  df = pd.read_excel(XLSX_PATH)
  for pid in SG_LABELS:
    index = df['subject'] == int(pid[3:5])
    age = int(df.loc[index, 'age'].values[0])
    gender = 'female' if df.loc[index, 'sex (F=1)'].values[0] == 1 else 'male'
    meta_dict[pid] = {'age': age, 'gender': gender}
  return meta_dict


def load_nebula_from_clouds(WORK_DIR, SG_LABELS, CHANNELS, TIME_RESOLUTION,
                            PROBE_KEYS, XLSX_PATH):
  freud = Freud(WORK_DIR)
  nebula = freud.load_nebula(sg_labels=SG_LABELS,
                             channels=CHANNELS,
                             time_resolution=TIME_RESOLUTION,
                             probe_keys=PROBE_KEYS)

  # (2.2) Set metadata
  meta_dict = load_sc_meta(XLSX_PATH, SG_LABELS)
  for pid in SG_LABELS: nebula.meta[pid] = meta_dict[pid]

  return nebula


def set_meta_to_nebula(nebula, XLSX_PATH) -> Nebula:
  meta_dict = load_sc_meta(XLSX_PATH, nebula.labels)
  for pid in nebula.labels: nebula.meta[pid] = meta_dict[pid]
  return nebula


def load_macro_and_meta_from_workdir(WORK_DIR, SG_LABELS, XLSX_PATH):
  x_dict = {}
  for pid in SG_LABELS:
    macro_path = os.path.join(WORK_DIR, pid, f'macro_alpha.od')
    assert os.path.exists(macro_path), f'Not found: {macro_path}'
    x_dict[pid] = io.load_file(macro_path)

  meta_dict = load_sc_meta(XLSX_PATH, SG_LABELS)
  return x_dict, meta_dict


CK_MAP = {
  'EEG Fpz-Cz': 'Fpz',
  'EEG Pz-Oz': 'Pz'
}

PK_MAP = {
  'FREQ-20': 'F', 'AMP-1': 'A', 'P-TOTAL': 'P', 'RP-DELTA': 'RD',
  'RP-THETA': 'RT', 'RP-ALPHA': 'RA', 'RP-BETA': 'RB',
  'KURT': 'KU', 'RPS-DELTA_THETA_AVG': 'RDT',
  'RPS-DELTA_ALPHA_AVG': 'RDA', 'RPS-THETA_ALPHA_AVG': 'RTA',
}

def get_joint_key(ck1, pk1, ck2, pk2):
  """
  CHANNELS = [ 'EEG Fpz-Cz', 'EEG Pz-Oz' ]

  PROBE_KEYS = [
    'FREQ-20', 'AMP-1', # 'GFREQ-35',
    'P-TOTAL', 'RP-DELTA', 'RP-THETA', 'RP-ALPHA', 'RP-BETA',
  ]
  """
  ck1, pk1, ck2, pk2 = CK_MAP[ck1], PK_MAP[pk1], CK_MAP[ck2], PK_MAP[pk2]
  if ck1 == ck2: return f'{ck1}_{pk1}x{pk2}'
  return f'({ck1}-{pk1})x({ck2}-{pk2})'
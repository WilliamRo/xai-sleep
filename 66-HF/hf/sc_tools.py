from hypnomics.freud.nebula import Nebula

import numpy as np



def get_paired_sg_labels(sg_labels: list, excludes=()):
  """e.g., label = `SC4012E`"""
  paired_sg_labels = []
  pids = [sg_label[2:5] for sg_label in sg_labels]
  for label, pid in zip(sg_labels, pids):
    if len([p for p in pids if p == pid]) > 1:
      if pid not in excludes: paired_sg_labels.append(label)

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
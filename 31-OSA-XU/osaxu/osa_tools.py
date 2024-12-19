from hypnomics.freud.freud import Freud
from roma import io

import os
import pandas as pd



def load_osa_meta(XLSX_PATH, SG_LABELS):
  meta_dict = {}
  df = pd.read_excel(XLSX_PATH)
  def get_value(i, k):
    values = df.loc[i, k].values
    if len(values) == 1: return float(values[0])
    elif len(values) == 0: return None
    else: raise ValueError(f"Multiple values found for {k}")

  for pid in SG_LABELS:
    index = df['pid'] == int(pid)

    def gv(k):
      val = get_value(index, k)
      return val

    meta_dict[pid] = {
      'age': gv('age'),
      'AHI': gv('AHI'),
      'gender': int(gv('gender') == 1),
      'BMI': gv('BMI'),
      'AHI': gv('AHI'),
      'REM_AHI': gv('REM_AHI'),
      'MMSE': gv('s_MMSE'),
      'cog_imp': gv('cog_imp'),
      'PHQ9': gv('s_PHQ9'),
      'dep': gv('dep'),
      'GAD7': gv('s_GAD7'),
      'anx': gv('anx'),
      'ESS': gv('s_ESS'),
      'som': gv('som'),
    }

  return meta_dict



def load_nebula_from_clouds(WORK_DIR, SG_LABELS, CHANNELS, TIME_RESOLUTION,
                            PROBE_KEYS, XLSX_PATH):
  freud = Freud(WORK_DIR)
  nebula = freud.load_nebula(sg_labels=SG_LABELS,
                             channels=CHANNELS,
                             time_resolution=TIME_RESOLUTION,
                             probe_keys=PROBE_KEYS)

  # Set metadata
  meta_dict = load_osa_meta(XLSX_PATH, SG_LABELS)
  for pid in SG_LABELS: nebula.meta[pid] = meta_dict[pid]

  return nebula



def load_macro_and_meta_from_workdir(WORK_DIR, SG_LABELS, XLSX_PATH):
  x_dict = {}
  for pid in SG_LABELS:
    macro_path = os.path.join(WORK_DIR, pid, f'macro_alpha.od')
    assert os.path.exists(macro_path), f'Not found: {macro_path}'
    x_dict[pid] = io.load_file(macro_path)

  meta_dict = load_osa_meta(XLSX_PATH, SG_LABELS)
  return x_dict, meta_dict



def set_target_collection_for_omix(omix, nebula):
  SG_LABELS = nebula.labels
  for key in nebula.meta[SG_LABELS[0]].keys():
    if key == 'gender': tl = ['Female', 'Male']
    elif key in ('cog_imp', 'dep', 'anx', 'som'): tl = ['Negative', 'Positive']
    else: tl = [key]
    omix.add_to_target_collection(
      key, [nebula.meta[pid][key] for pid in nebula.labels], target_labels=tl)

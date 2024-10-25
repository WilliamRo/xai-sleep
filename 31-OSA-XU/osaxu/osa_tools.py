from hypnomics.freud.freud import Freud

import pandas as pd



def load_nebula_from_clouds(WORK_DIR, SG_LABELS, CHANNELS, TIME_RESOLUTION,
                            PROBE_KEYS, XLSX_PATH):
  freud = Freud(WORK_DIR)
  nebula = freud.load_nebula(sg_labels=SG_LABELS,
                             channels=CHANNELS,
                             time_resolution=TIME_RESOLUTION,
                             probe_keys=PROBE_KEYS)

  # Set metadata
  df = pd.read_excel(XLSX_PATH)
  def get_value(i, k):
    values = df.loc[i, k].values
    if len(values) == 1: return float(values[0])
    elif len(values) == 0: return None
    else: raise ValueError(f"Multiple values found for {k}")

  for pid in nebula.labels:
    index = df['pid'] == int(pid)

    def gv(k):
      val = get_value(index, k)
      return val

    nebula.meta[pid] = {
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

  return nebula

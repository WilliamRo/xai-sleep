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
  get_value = lambda i, k: float(df.loc[i, k].values[0])

  for pid in nebula.labels:
    index = df['Night code'] == '01-' + pid
    AHI = get_value(index, 'AHI')
    MA = get_value(index, 'MA')
    PLMS = get_value(index, 'PLMS')
    PLMS_A = get_value(index, 'PLMS-A')
    REF = 0 if df.loc[index, 'Montage reference'].values[0] == 'CLE' else 1

    nebula.meta[pid] = {'AHI': AHI, 'MA': MA, 'PLMS': PLMS, 'PLMS-A': PLMS_A,
                        'reference': REF}

  return nebula

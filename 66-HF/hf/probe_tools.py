import os



def get_probe_keys(PROBE_CONFIG):
  PROBE_KEYS = []

  # (1.4.1) Part A
  if 'A' in PROBE_CONFIG: PROBE_KEYS.extend(['FREQ-20', 'AMP-1'])

  # (1.4.2) Part B
  if 'B' in PROBE_CONFIG: PROBE_KEYS.extend(
    ['P-TOTAL', 'RP-DELTA', 'RP-THETA', 'RP-ALPHA', 'RP-BETA'])

  # (1.4.3) Part C (sun2017)
  if 'C' in PROBE_CONFIG:
    # (1.4.3.1)
    PROBE_KEYS.extend(['MAG', 'KURT', 'ENTROPY'])

    # (1.4.3.2)
    for b1, b2 in [('DELTA', 'TOTAL'), ('THETA', 'TOTAL'), ('ALPHA', 'TOTAL'),
                   ('DELTA', 'THETA'), ('DELTA', 'ALPHA'), ('THETA', 'ALPHA')]:
      for stat_key in ['95', 'MIN', 'AVG', 'STD']:
        PROBE_KEYS.append(f'RPS-{b1}_{b2}_{stat_key}')

    # (1.4.3.3)
    for b in ['DELTA', 'THETA', 'ALPHA', 'SIGMA']:
      PROBE_KEYS.append(f'BKURT-{b}')

  # (1.4.4) Part D (a useful part of sun2017)
  if 'D' in PROBE_CONFIG:
    assert 'C' not in PROBE_CONFIG
    PROBE_KEYS.extend(['KURT'])

    for b1, b2 in [('DELTA', 'THETA'), ('DELTA', 'ALPHA'), ('THETA', 'ALPHA')]:
      PROBE_KEYS.append(f'RPS-{b1}_{b2}_AVG')

  return PROBE_KEYS

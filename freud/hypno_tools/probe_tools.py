from collections import OrderedDict
from hypnomics.hypnoprints.probes import ProbeLibrary



def get_extractor_dict(keys, **kwargs):
  od = OrderedDict()

  for key in keys:
    if '-' in key: name, arg = key.split('-')
    else: name, arg = key, None

    if name == 'AMP':
      fs = kwargs.get('fs')
      ws = float(arg)
      od[key] = lambda s, fs=fs, ws=ws: ProbeLibrary.amplitude(
        s, fs=fs, window_size=ws)
    elif name == 'FREQ':
      fs = kwargs.get('fs')
      fmax = float(arg)
      od[key] = lambda s, fs=fs, fmax=fmax: ProbeLibrary.frequency_stft(
        s, fs=fs, fmax=fmax)
    elif name == 'GFREQ':
      fs = kwargs.get('fs')
      fmax = float(arg)
      od[key] = lambda s, fs=fs, fmax=fmax: ProbeLibrary.frequency_st(
        s, fs=fs, fmax=fmax)
    elif name in ('P', 'RP'):
      fs = kwargs.get('fs')
      od[key] = lambda s, fs=fs, band=arg: ProbeLibrary.total_power(
        s, fs=fs, band=band)
    elif name == 'MAG':
      od[key] = lambda s: ProbeLibrary.mean_absolute_gradient(s)
    elif name == 'KURT':
      od[key] = lambda s: ProbeLibrary.kurtosis(s)
    elif name == 'ENTROPY':
      od[key] = lambda s: ProbeLibrary.sample_entropy(s)
    elif name == 'RPS':
      fs = kwargs.get('fs')
      b1, b2, st = arg.split('_')
      st = {'95': '95th percentile', 'MIN': 'min',
            'AVG': 'mean', 'STD': 'std'}[st]
      od[key] = lambda s, fs=fs, b1=b1, b2=b2, st=st: ProbeLibrary.relative_power_stats(
        s, fs=fs, band1=b1, band2=b2, stat_key=st)
    elif name == 'BKURT':
      fs = kwargs.get('fs')
      od[key] = lambda s, fs=fs, band=arg: ProbeLibrary.band_kurtosis(
        s, fs=fs, band=band)
    else: raise KeyError(f'Unknown key: {key}')

  return od



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

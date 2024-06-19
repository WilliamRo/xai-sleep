from collections import OrderedDict
from hypnomics.hypnoprints.probes import ProbeLibrary



def get_extractor_dict(keys, **kwargs):
  od = OrderedDict()

  for key in keys:
    name, arg = key.split('-')
    if name == 'AMP':
      fs = kwargs.get('fs')
      ws = float(arg)
      od[key] = lambda s, fs=fs, ws=ws: ProbeLibrary.amplitude(
        s, fs=fs, window_size=ws)
    elif name == 'FREQ':
      fs = kwargs.get('fs')
      fmax = float(arg)
      od[key] = lambda s, fs=fs, fmax=fmax: ProbeLibrary.frequency(
        s, fs=fs, fmax=fmax)
    else: raise KeyError(f'Unknown key: {key}')

  return od

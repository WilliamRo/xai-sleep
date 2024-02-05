from epoch_explorer_base import EpochExplorer
from epoch_explorer_omic import RhythmPlotterPro
from matplotlib.patches import Rectangle
from matplotlib.widgets import RectangleSelector
from pictor import Pictor
from pictor.plotters.plotter_base import Plotter
from pictor.objects.signals.signal_group import SignalGroup, Annotation
from pictor.plugins import DialogUtilities
from roma import console

import matplotlib.pyplot as plt
import numpy as np
import os



class RhythmWalker(RhythmPlotterPro):

  def walk_on_selected_sg(self, sg_list, channels, biomarkers, bm_args,
                          save_path):
    """Yields `results`: dict, where
       - results['meta'] = (
           sg_label_list,
           channel_list,
           bm_args_dict     # {'BM01': ('arg1', [32, 64]), ...}
         )
       - results[(<sg_label>, <channel>, (<bm_key>, <arg>, <value>)] =
         {<stage_key>: bm_output_list}

       Currently, at most one argument is allowed for each BM.
    """
    results = {}
    # (1) Fill-in meta
    meta = ([sg.label for sg in sg_list], channels, bm_args)
    results['meta'] = meta

    # (2) Calculate probing results
    n, N = -1, len(sg_list) * len(channels)
    for i, sg in enumerate(sg_list):
      assert isinstance(sg, SignalGroup)
      console.show_status(f'Analyzing {sg.label} ...')
      channels_names = sg.digital_signals[0].channels_names
      for channel_key in channels:
        n += 1
        console.show_status(f'Walking through `{sg.label}[{channel_key}]` ...')
        for bm_key, bm_func in biomarkers.items():
          console.print_progress(n, N)
          arg_key, arg_list = bm_args[bm_key]
          for arg in arg_list:
            res_key = (sg.label, channel_key, (bm_key, arg_key, arg))
            config = {arg_key: arg} if arg_key else {}
            # Get signal
            se: dict = self.explorer.get_sg_stage_epoch_dict(sg)
            res_dict = {}
            for stage_key, data_list in se.items():
              signals = [data[:, channels_names.index(channel_key)]
                         for data in data_list]
              res_dict[stage_key] = self.calculate_biomarkers(
                bm_func, signals, **config)
            results[res_key] = res_dict

    console.show_status(f'Walking through {len(sg_list)} signal groups ...')

    # Save
    from tframe.utils.file_tools.io_utils import save
    save(results, save_path)
    console.show_status(f'Results saved to {save_path}.')

    return results

  # region: Scripts

  @staticmethod
  def lins(x1, x2, step, dtype=int):
    num = (x2 - x1) // step + 1
    return np.linspace(x1, x2, num=num, dtype=dtype)

  def export_000(self):
    sg_list = self.explorer.axes[self.explorer.Keys.OBJECTS][:5]
    channels = ['EEG Fpz-Cz', 'EEG Pz-Oz']
    biomarkers = {'BM01-FREQ': self._bm_01_freq, 'BM02-AMP': self._bm_02_ampl}
    args = {'BM01-FREQ': ('max_freq', self.lins(15, 35, 10)),
            'BM02-AMP': ('pool_size', self.lins(32, 224, 96))}

    save_file_name = 'sg10_eeg2_bm01(15,35)_bm02(32,224).pr'
    save_path = os.path.join(r'P:/xai-sleep/data/probe_reports',
                             save_file_name)

    self.walk_on_selected_sg(sg_list, channels, biomarkers, args, save_path)
  e0 = export_000

  def export_001(self):
    sg_list = self.explorer.axes[self.explorer.Keys.OBJECTS][:10]
    channels = ['EEG Fpz-Cz', 'EEG Pz-Oz']
    biomarkers = {'BM01-FREQ': self._bm_01_freq, 'BM02-AMP': self._bm_02_ampl}
    args = {'BM01-FREQ': ('max_freq', self.lins(15, 40, 5)),
            'BM02-AMP': ('pool_size', self.lins(32, 256, 32))}

    save_file_name = 'sg10_eeg2_bm01(15,40)_bm02(32,256).pr'
    save_path = os.path.join(r'P:/xai-sleep/data/probe_reports',
                             save_file_name)

    self.walk_on_selected_sg(sg_list, channels, biomarkers, args, save_path)

  def export_002(self):
    sg_list = self.explorer.axes[self.explorer.Keys.OBJECTS]
    channels = ['EEG Fpz-Cz', 'EEG Pz-Oz']
    biomarkers = {'BM01-FREQ': self._bm_01_freq, 'BM02-AMP': self._bm_02_ampl}
    args = {'BM01-FREQ': ('max_freq', [25]),
            'BM02-AMP': ('pool_size', [128])}

    save_file_name = 'sg10_eeg2_bm01(25)_bm02(128).pr'
    save_path = os.path.join(r'P:/xai-sleep/data/probe_reports',
                             save_file_name)

    self.walk_on_selected_sg(sg_list, channels, biomarkers, args, save_path)

  # endregion: Scripts



if __name__ == '__main__':
  from roma import finder
  from roma import io

  # Set directories
  data_dir = r'../../data/'
  # data_dir += 'sleepeasonx'
  # data_dir += 'sleepedfx'
  data_dir += 'rrsh-osa'

  prefix = ['', 'sleepedfx', 'ucddb', 'rrsh'][1]
  pattern = f'{prefix}*.sg'
  # pattern = f'SC*raw*.sg'

  # For rrsh-osa
  pattern = f'*(trim;easy;100).sg'

  channel_names = ['EEG Fpz-Cz', 'EEG Pz-Oz']

  # Select .sg files
  sg_file_list = finder.walk(data_dir, pattern=pattern)[:20]

  signal_groups = []
  for path in sg_file_list:
    sg: SignalGroup = io.load_file(path, verbose=True)
    if channel_names: sg = sg.extract_channels(channel_names)
    signal_groups.append(sg)

  # Visualize signal groups
  EpochExplorer.explore(signal_groups, plot_wave=True,
                        plotter_cls=RhythmWalker)



from collections import OrderedDict
from hypnomics.freud.freud import Freud
from roma import finder
from roma import io
from sc.fp_viewer import FPViewer

import os



def view_fingerprints(cloud_dir, sg_label_list,
                      channels: list, time_resolution=30,
                      pk1='FREQ-20', pk2='AMP-1', **configs):
  # (1) Get version 1 finger prints
  fps = OrderedDict()

  probe_dict = OrderedDict()
  key1, arg1 = pk1.split('-')
  key2, arg2 = pk2.split('-')
  probe_dict[key1] = ('a', [arg1])
  probe_dict[key2] = ('a', [arg2])

  fps['meta'] = (sg_label_list, channels, probe_dict)

  freud = Freud(cloud_dir)

  for sg_label in sg_label_list:
    for chn in channels:
      pk1_path, b_exist = freud._check_hierarchy(
        sg_label, chn, time_resolution, pk1, create_if_not_exist=False)
      assert b_exist, f'`{pk1_path}` not found.'
      key = (sg_label, chn, (key1, 'a', arg1))
      fps[key] = io.load_file(pk1_path)

      pk2_path, b_exist = freud._check_hierarchy(
        sg_label, chn, time_resolution, pk2, create_if_not_exist=False)
      assert b_exist, f'`{pk2_path}` not found.'
      key = (sg_label, chn, (key2, 'a', arg2))
      fps[key] = io.load_file(pk2_path)

  # (2) Show fingerprints
  fpv = FPViewer(walker_results=fps)
  for k, v in configs.items(): fpv.plotters[0].set(k, v)
  fpv.show()


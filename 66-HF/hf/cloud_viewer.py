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
  freud = Freud(cloud_dir)
  nebula = freud.load_nebula(sg_labels=sg_label_list, channels=channels,
                             time_resolution=time_resolution,
                             probe_keys=[pk1, pk2])
  fps = nebula.to_walker_results(x_key=pk1, y_key=pk2)

  # (2) Show fingerprints
  fpv = FPViewer(walker_results=fps, title=f'Clouds delta={time_resolution}')
  for k, v in configs.items(): fpv.plotters[0].set(k, v)
  fpv.show()


from freud.talos_utils.slp_set import SleepSet
from pictor.objects.signals.signal_group import SignalGroup, DigitalSignal
from pictor.objects.signals.signal_group import Annotation
from roma.spqr.finder import walk
from roma import io
from tframe import console
from typing import List

import os
import numpy as np



class SleepEason(SleepSet):
  FILE_LIST_KEY = 'file_list'

  def __init__(self, data_dir, buffer_size=None):
    """buffer_size decides how many files to fetch per round
    """
    self.buffer_size = buffer_size
    self.data_dict = {}

    self.properties = {}
    self.properties[self.FILE_LIST_KEY] = walk(data_dir, pattern='*.sg')

    self.data_fetcher = self.fetch_data

  @property
  def file_list(self): return self.properties[self.FILE_LIST_KEY]

  def fetch_data(self):
    if self.buffer_size is None: files = self.file_list
    else: files = np.random.choice(
      self.file_list, self.buffer_size, replace=False)

    console.show_status('Fetching signal groups ...')
    self.signal_groups = []
    for p in files:
      sg = io.load_file(p)
      console.supplement(f'Loaded `{p}`', level=2)
      self.signal_groups.append(sg)

  @classmethod
  def load_as_sleep_set(cls, data_dir, **kwargs):
    from tframe import hub as th

    return SleepEason(data_dir, buffer_size=th.sg_buffer_size)



if __name__ == '__main__':
  from pprint import pprint

  data_dir = r'../../../data/eason-alpha'

  se = SleepEason(data_dir, buffer_size=10)
  se.fetch_data()

  # pprint(se.properties)

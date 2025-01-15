from collections import OrderedDict
from datetime import datetime
from freud.talos_utils.slp_set import SleepSet
from roma import console, io, Nomear
from pictor.objects.signals.signal_group import Annotation
from pictor.objects.signals.signal_group import DigitalSignal, SignalGroup
from typing import List

import os
import numpy as np
import math
import re
import traceback



class SHHSet(SleepSet):
  """Reference: ...

  Dataset Folder Structure
  ------------------------
  ...
  """

  ANNO_LABELS = ['Sleep_stage_W', 'Sleep_stage_N1', 'Sleep_stage_N2',
                 'Sleep_stage_N3', 'Sleep_stage_R',  'Sleep_stage_?']

  EEG_EOG = ['F3-M2', 'F4-M1', 'C3-M2', 'C4-M1', 'O1-M2', 'O2-M1',
             'E1-M[12]', 'E2-M[12]']

  GROUPS = [('EEG F3-M2', 'EEG C3-M2', 'EEG O1-M2',
             'EEG F4-M1', 'EEG C4-M1', 'EEG O2-M1')]

  @staticmethod
  def channel_map(edf_ck):
    """Map EDF channel names to standard channel names. Used in reading raw data
    """
    # For edf_ck match 'F3-M2', 'F4-M1', 'C3-M2', 'C4-M1', 'O1-M2', 'O2-M1'
    # Using regular expression
    if re.match(r'^[FCO][1234][\-]M[12]$', edf_ck):
      return f'EEG {edf_ck}'

    # In some cases, two EOG channels may use the same reference electrode
    # e.g., E1-M2, E2-M2 (sub-S0001111190905_ses-1)
    #       E1-M2, E2-M1 (sub-S0001111190905_ses-4)
    if re.match(r'^E[12][\-]M[12]$', edf_ck):
      return f'EOG {edf_ck}'

    return edf_ck

  # region: Data Conversion

  @classmethod
  def load_sg_from_raw_files(cls, ses_dir, dtype=np.float16, max_sfreq=128,
                             bipolar=False, **kwargs):
    """Convert an `.edf` file into a SignalGroup.
    """
    sg = None
    return sg


  @classmethod
  def load_shhs_annotation(cls, anno_path):
    """
    """
    import pandas as pd


  @classmethod
  def convert_rawdata_to_signal_groups(
      cls, ses_folder_list: list, tgt_dir, dtype=np.float16, max_sfreq=128,
      bipolar=False, **kwargs):

    pass

  # endregion: Data Conversion



class SHHSAgent(Nomear):
  """
  """

  def __init__(self, meta_dir, data_dir=None, meta_version='0.21.0'):
    self.meta_dir = meta_dir
    self.meta_version = meta_version
    self.data_dir = data_dir

  # region: Properties

  @property
  def meta_path(self): return os.path.join(
    self.meta_dir, f'shhs-harmonized-dataset-{self.meta_version}.csv')

  # endregion: Properties

  # region: Public Methods

  # endregion: Public Methods



if __name__ == '__main__':
  import pandas as pd

  OVERWRITE = 1


  print('Done!')

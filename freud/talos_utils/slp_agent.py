from typing import List
from freud.talos_utils.slp_set import SleepSet
from tframe import console
from tframe.data.base_classes import DataAgent

import os



class SleepAgent(DataAgent):
  """Load sleep data according to `th.data_config` whose syntax is
      '<data_name> (configs)*'
  Here, <data_name> should be one of
  (1) 'sleepedfx'; (2) 'ucddb'; (3) 'rrshv1'
  """

  @classmethod
  def load_data(cls) -> List[SleepSet]:
    return []


  @classmethod
  def load_as_tframe_data(cls, th, **kwargs) -> SleepSet:
    """"""
    from freud.talos_utils.slp_config import SleepConfig
    assert isinstance(th, SleepConfig)

    data_name = th.data_name.lower()
    if data_name == 'sleepedfx':
      from freud.talos_utils.sleep_sets.sleepedfx import SleepEDFx as DSClass
      folder_name = 'sleepedf'
    elif data_name == 'ucddb':
      from freud.talos_utils.sleep_sets.ucddb import UCDDB as DSClass
      folder_name = 'ucddb'
    elif data_name == 'rrshv1':
      from freud.talos_utils.sleep_sets.rrshv1 import RRSHSCv1 as DSClass
      folder_name = 'rrsh'
    else: raise KeyError(f'!! Unknown data set `{data_name}`')

    data_dir = os.path.join(th.data_dir, folder_name)
    ds = DSClass.load_as_sleep_set(data_dir)

    return ds



if __name__ == '__main__':
  from freud.talos_utils.slp_config import SleepConfig as Hub

  th = Hub(as_global=True)

  th.data_config = 'sleepedfx'
  th.data_dir = r'../../data/'

  ds = SleepAgent.load_as_tframe_data(th)
  ds.show()


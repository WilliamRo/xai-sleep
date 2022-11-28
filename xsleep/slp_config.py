from tframe.trainers import SmartTrainerHub
from tframe.configs.config_base import Flag

from roma import console


class SLPConfig(SmartTrainerHub):

  class Datasets:
    ucddb = 'ucddb'

  report_detail = Flag.boolean(False, 'Whether to report detail')
  partition_over_patients = Flag.boolean(True, '...')

  show_in_monitor = Flag.boolean(False, 'Whether to show results in monitor')
  predictions = Flag.list([], 'the index of predicted data')

  window_size = Flag.float(35, 'The number of epoch in a sample')

  use_gate = Flag.boolean(False, 'Replace correct data with unknown data')
  test_config = Flag.string(None, 'the setting of cross validation', is_key=None)
  ratio = Flag.float(0.1, 'The number of epoch in a sample', is_key=None)


# New hub class inherited from SmartTrainerHub must be registered
SLPConfig.register()


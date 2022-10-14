from tframe.trainers import SmartTrainerHub
from tframe.configs.config_base import Flag

from roma import console


class SLPConfig(SmartTrainerHub):

  class Datasets:
    ucddb = 'ucddb'

  report_detail = Flag.boolean(False, 'Whether to report detail')
  partition_over_patients = Flag.boolean(True, '...')



# New hub class inherited from SmartTrainerHub must be registered
SLPConfig.register()


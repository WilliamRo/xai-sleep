from tframe.trainers import SmartTrainerHub
from tframe.configs.config_base import Flag



class SLPConfig(SmartTrainerHub):
  pass



# New hub class inherited from SmartTrainerHub must be registered
SLPConfig.register()

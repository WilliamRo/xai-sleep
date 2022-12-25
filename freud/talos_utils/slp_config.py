from roma import Arguments
from tframe.trainers import SmartTrainerHub
from tframe.configs.config_base import Flag



class SleepConfig(SmartTrainerHub):

  @property
  def data_name(self):
    return Arguments.parse(self.data_config).func_name

  @property
  def data_args(self):
    return Arguments.parse(self.data_config).arg_list

  @property
  def data_kwargs(self) -> dict:
    return Arguments.parse(self.data_config).arg_dict



# New hub class inherited from SmartTrainerHub must be registered
SleepConfig.register()

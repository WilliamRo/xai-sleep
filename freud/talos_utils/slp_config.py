from roma import Arguments
from tframe.trainers import SmartTrainerHub
from tframe.configs.config_base import Flag



class SleepConfig(SmartTrainerHub):
  """
  th.data_config format:
  <data-name> <channels> [other-setting]
  """

  balance_training_stages = Flag.boolean(
    True, 'Whether to balance training stages', is_key=None)
  epoch_delta = Flag.float(
    0.0, 'Delta for generating batches, should be in [0, 1)', is_key=None)
  use_gen_batches_buffer = Flag.boolean(
    False, 'Whether to use gen_batches buffer')

  # region: Properties

  @property
  def data_name(self):
    return Arguments.parse(self.data_config).func_name

  @property
  def data_args(self):
    return Arguments.parse(self.data_config).arg_list

  @property
  def data_kwargs(self) -> dict:
    return Arguments.parse(self.data_config).arg_dict

  @property
  def fusion_channels(self):
    return [s.split(',') for s in self.data_args[0].split(';')]

  # endregion: Properties



# New hub class inherited from SmartTrainerHub must be registered
SleepConfig.register()

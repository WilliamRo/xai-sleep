from pictor.objects.signals.signal_group import DigitalSignal, SignalGroup
from tframe.data.sequences.seq_set import SequenceSet
from typing import List



class SleepSet(SequenceSet):

  # region: Properties

  @property
  def signal_groups(self) -> List[SignalGroup]:
    return self.properties['signal_groups']

  # endregion: Properties

  # region: Overwriting

  def _check_data(self): pass

  # endregion: Overwriting

  # region: Abstract Methods

  @classmethod
  def load_as_signal_groups(cls, data_dir, **kwargs) -> List[SignalGroup]:
    raise NotImplementedError

  @classmethod
  def load_as_sleep_set(cls, data_dir, **kwargs) -> SequenceSet:
    raise NotImplementedError

  # endregion: Abstract Methods

  # region: Data Reading



  # endregion: Data Reading



if __name__ == '__main__':
  from roma import console

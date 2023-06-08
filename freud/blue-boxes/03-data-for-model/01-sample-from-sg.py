from tframe import console
from freud.talos_utils.sleep_sets.sleepedfx import SleepEDFx, SleepSet
from freud.talos_utils.slp_config import SleepConfig



console.suppress_logging()
th = SleepConfig(as_global=True)
th.data_config = 'sleepedfx 1,2'

data_dir = r'../../../data/sleepedfx-mini'

preprocess = 'trim;iqr'
ds: SleepSet = SleepEDFx.load_as_sleep_set(
  data_dir, overwrite=0, preprocess=preprocess)

# Configure ds so that tapes can be extracted according to th.data_config
ds.configure()

sg = ds.signal_groups[0]
# data, labels = ds._sample_seqs_from_sg(sg, 0, 60, with_stage=True)

th.epoch_num = 10
ds = ds._get_sequence_randomly(2)


print()

from freud.talos_utils.slp_set import SleepSet, DataSet
from pictor.objects.signals.signal_group import SignalGroup, Annotation
from s2s_core import th, du, console



# Load data
th.data_config = 'sleepeason1 1,2'
th.data_config += ' val_ids=16,17 test_ids=18,19'

train_set, val_set, test_set = du.load_data()
ds = test_set

# Investigate signal groups
console.show_info('Annotations in Signal Groups')
signal_groups = ds.properties['signal_groups']

sg: SignalGroup = signal_groups[0]
anno: Annotation = list(sg.annotations.values())[0]
labels = anno.labels
stage_nums = [0 for l in labels]

for sg in signal_groups:
  anno: Annotation = list(sg.annotations.values())[0]
  for (t1, t2), s in zip(anno.intervals, anno.annotations):
    stage_nums[s] += (t2 - t1) // 30

for lb, num in zip(labels, stage_nums):
  console.supplement(f'{lb}: {int(num)}', level=2)





from freud.talos_utils.sleep_sets.rrshv1 import RRSHSCv1
from tframe import console



def print_stages(path):
  signal_groups = RRSHSCv1.load_as_signal_groups(path)
  console.show_info(f'{path}:')

  for sg in signal_groups:
    anno = sg.annotations[RRSHSCv1.ANNO_KEY_GT_STAGE]
    stages = anno.annotations
    labels = ['W', '1', '2', '3', 'R']
    brief = [stages[0]]
    for s in stages:
      if s != brief[-1] and s < 5: brief.append(s)
    brief = [labels[i] for i in brief if i < 5]
    console.supplement(brief, level=2)

console.suppress_logging()
data_dir = [
  r'../../data/rrsh',
  r'../../data/rrsh-nacrolepsy',
]
print_stages(data_dir[0])
print_stages(data_dir[1])


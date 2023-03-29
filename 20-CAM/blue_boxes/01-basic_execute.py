from freud.talos_utils.sleep_sets.sleepedfx import SleepEDFx, SleepSet
from tframe import console
from cam_oliver.oliver import Oliver



console.suppress_logging()
data_dir = r'../../data/sleepedfx'
preprocess = 'trim;iqr'
signal_groups = SleepEDFx.load_as_signal_groups(data_dir, preprocess=preprocess)

oliver = Oliver(title='Oliver')
oliver.objects = signal_groups
oliver.ssa()

t_path = r'E:\wanglin\project\deep_learning\xai-sleep\20-CAM\01_cam\checkpoints\0322_cam(48-48-m-32-32-m-16)\0322_cam(48-48-m-32-32-m-16).py'
oliver.monitor.stage('1,2', t_path, grad_cam=True)
oliver.monitor.set('max_ticks', None)
oliver.show()

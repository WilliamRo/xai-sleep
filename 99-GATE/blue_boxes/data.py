from gate_core import th
from slp_datasets.sleepedfx import SleepEDFx


th.use_gate = True
th.data_config = 'sleepedf:20:0,1,2'
th.ratio = 0
th.overwrite = True
th.show_in_monitor = False
data_name, data_num, channel_select = th.data_config.split(':')
data_set = SleepEDFx.load_as_tframe_data(th.data_dir,
                                         data_name,
                                         data_num,
                                         suffix='-alpha')
data_set.configure(channel_select=channel_select)
data_set.report()
data_set.show()

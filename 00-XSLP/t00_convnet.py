import xslp_core as core
import xslp_mu as m

from tframe import console
from tframe import tf
from tframe.utils.misc import date_string
from tframe.utils.organizer.task_tools import update_job_dir

# -----------------------------------------------------------------------------
# Define model here
# -----------------------------------------------------------------------------
model_name = 'Convnet'
id = 2


def model(): return m.get_model()

def main(_):
  console.start('{} on sleep stage task'.format(model_name.upper()))

  th = core.th
  # ---------------------------------------------------------------------------
  # 0. date set setup
  # ---------------------------------------------------------------------------
  th.data_config = 'sleepedf:20:0,4'

  if 'apnea' in th.data_config:
    th.output_dim = 2
    th.input_shape = [120, 1]
    th.val_size = 5
    th.test_size = 35
  else:
    th.output_dim = 5
    channel_num = len(th.data_config.split(':')[2].split(','))
    th.input_shape = [3000, channel_num]

  # ---------------------------------------------------------------------------
  # 1. folder/file names and device
  # ---------------------------------------------------------------------------
  update_job_dir(id, model_name)
  summ_name = model_name
  th.prefix = '{}_'.format(date_string())

  th.visible_gpu_id = 0
  # ---------------------------------------------------------------------------
  # 2. model setup
  # ---------------------------------------------------------------------------
  th.model = model

  th.activation = 'relu'
  th.use_batchnorm = True
  # ---------------------------------------------------------------------------
  # 3. trainer setup
  # ---------------------------------------------------------------------------
  th.epoch = 1
  th.batch_size = 32

  th.optimizer = 'adam'
  th.learning_rate = 0.0001
  th.train = True
  th.overwrite = True
  th.print_cycle = 10
  th.save_model = True


  # ---------------------------------------------------------------------------
  # 4. other stuff and activate
  # ---------------------------------------------------------------------------
  th.mark = '{}({})'.format(model_name, th.data_config.split(':')[0])
  th.gather_summ_name = th.prefix + summ_name + '.sum'
  core.activate()


if __name__ == '__main__':
  console.suppress_logging()
  tf.app.run()

import uslp_core as core
import uslp_mu as m

from tframe import console
from tframe import tf
from tframe.utils.misc import date_string
from tframe.utils.organizer.task_tools import update_job_dir

# -----------------------------------------------------------------------------
# Define model here
# ----------------------------------------------------------------------------
model_name = 'U-sleep'
id = 2


def main(_):
  console.start('{} on sleep stage task'.format(model_name.upper()))

  th = core.th
  # ---------------------------------------------------------------------------
  # 0. date set setup
  # ---------------------------------------------------------------------------
  th.data_config = 'sleepedf:20:0,2'
  th.output_dim = 5
  th.input_shape = [105000, 2]

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
  th.activation = 'relu'
  th.use_batchnorm = True
  th.train = True

  # ---------------------------------------------------------------------------
  # 3. trainer setup
  # ---------------------------------------------------------------------------
  th.epoch = 1000
  th.batch_size = 64
  th.optimizer = 'adam'
  th.learning_rate = 0.0000001
  th.overwrite = True
  th.print_cycle = 10
  th.save_model = True
  # per sample length
  th.window_size = 35
  th.random_sample_length = 3000

  # ---------------------------------------------------------------------------
  # 4. other stuff and activate
  # ---------------------------------------------------------------------------
  th.mark = '{}({})'.format(model_name, th.data_config.split(':')[0])
  th.gather_summ_name = th.prefix + summ_name + '.sum'

  th.model = m.get_model
  core.activate()


if __name__ == '__main__':
  console.suppress_logging()
  tf.app.run()

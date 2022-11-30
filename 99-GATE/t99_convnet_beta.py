import gate_core as core
import gate_mu as m

from tframe import console
from tframe import tf
from tframe.utils.misc import date_string
from tframe.utils.organizer.task_tools import update_job_dir

# -----------------------------------------------------------------------------
# Define model here
# -----------------------------------------------------------------------------
model_name = 'feature_fusion'
id = 2


def model(): return m.get_feature_fusion_model()


def main(_):
  console.start('{} on sleep stage task'.format(model_name.upper()))

  th = core.th
  # ---------------------------------------------------------------------------
  # 0. date set setup
  # ---------------------------------------------------------------------------
  th.data_config = 'sleepedf:20:0,1,2'

  th.output_dim = 5
  th.input_shape = [3000, 3]

  # --------------------------------------------------------------------------
  # 1. folder/file names and device
  # ---------------------------------------------------------------------------
  update_job_dir(id, model_name)
  summ_name = model_name
  th.prefix = '{}_'.format(date_string())

  th.visible_gpu_id = 0
  # -------------------;a--------------------------------------------------------
  # 2. model setup
  # ---------------------------------------------------------------------------
  th.channels = '0,1;2'
  th.kernel_size = 3
  th.activation = 'relu'
  th.use_batchnorm = True

  th.model = model
  # ---------------------------------------------------------------------------
  # 3. trainer setup
  # ---------------------------------------------------------------------------
  th.epoch = 1000
  th.batch_size = 32
  th.dropout = 0.5
  th.archi_string = '4-8-m-16-24-m-64'
  th.optimizer = 'adam'
  th.learning_rate = 0.0001

  th.train = True
  th.overwrite = True
  th.use_gate = True
  th.ratio = 0.3
  th.test_config = 'test-data:0,1'
  th.show_in_monitor = True

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

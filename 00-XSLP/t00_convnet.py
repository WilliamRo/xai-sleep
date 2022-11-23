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


def model():
  th = core.th
  # th.developer_code = 'expand'
  model = m.get_container(flatten=False)
  fm = m.mu.ForkMergeDAG(vertices=[
    [m.mu.Conv1D(filters=64, kernel_size=50, use_batchnorm=th.use_batchnorm,
                 strides=6, activation=th.activation),
     m.mu.MaxPool1D(pool_size=8, strides=8), m.mu.Dropout(0.5),
     m.mu.Conv1D(filters=128, kernel_size=4, use_batchnorm=th.use_batchnorm,
                 activation=th.activation),
     m.mu.Conv1D(filters=128, kernel_size=4, use_batchnorm=th.use_batchnorm,
                 activation=th.activation),
     m.mu.Conv1D(filters=128, kernel_size=4, use_batchnorm=th.use_batchnorm,
                 activation=th.activation),
     m.mu.Conv1D(filters=128, kernel_size=4, use_batchnorm=th.use_batchnorm,
                 activation=th.activation),
     m.mu.MaxPool1D(pool_size=8, strides=8)],
    [m.mu.Conv1D(filters=64, kernel_size=400,
                 use_batchnorm=th.use_batchnorm,
                 strides=50, activation=th.activation),
     m.mu.MaxPool1D(pool_size=4, strides=4), m.mu.Dropout(0.5),
     m.mu.Conv1D(filters=128, kernel_size=8, use_batchnorm=th.use_batchnorm,
                 activation=th.activation),
     m.mu.Conv1D(filters=128, kernel_size=8, use_batchnorm=th.use_batchnorm,
                 activation=th.activation),
     m.mu.Conv1D(filters=128, kernel_size=8, use_batchnorm=th.use_batchnorm,
                 activation=th.activation),
     m.mu.Conv1D(filters=128, kernel_size=8, use_batchnorm=th.use_batchnorm,
                 activation=th.activation),
     m.mu.MaxPool1D(pool_size=2, strides=2)],
    [m.mu.Merge.Sum(), m.mu.Dropout(0.5)],
    [m.mu.Conv1D(filters=32, kernel_size=6, use_batchnorm=th.use_batchnorm,
                 activation=th.activation, dilation_rate=5),
     m.mu.Conv1D(filters=64, kernel_size=6, use_batchnorm=th.use_batchnorm,
                 activation=th.activation, dilation_rate=5),
     m.mu.Conv1D(filters=128, kernel_size=6, use_batchnorm=th.use_batchnorm,
                 activation=th.activation, dilation_rate=5),
     m.mu.Dropout(0.5)],
    [m.mu.Merge.Sum(), m.mu.Dropout(0.5)]],
    edges='1;10;011;0001;00011')
  model.add(fm)
  # endregion
  # Add flatten layer
  model.add(m.mu.Flatten())
  return m.finalize(model)


def main(_):
  console.start('{} on sleep stage task'.format(model_name.upper()))

  th = core.th
  # ---------------------------------------------------------------------------
  # 0. date set setup
  # ---------------------------------------------------------------------------
  th.data_config = 'rrsh::1,2,6'

  if 'apnea' in th.data_config:
    th.output_dim = 2
    th.input_shape = [120, 1]
    th.val_size = 5
    th.test_size = 35
  else:
    th.output_dim = 5
    th.input_shape = [3000, 3]

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
  th.epoch = 1000
  th.batch_size = 32

  th.optimizer = 'adam'
  th.learning_rate = 0.0001

  th.train = True
  th.overwrite = True
  th.print_cycle = 10
  th.save_model = True

  th.rehearse = True

  # ---------------------------------------------------------------------------
  # 4. other stuff and activate
  # ---------------------------------------------------------------------------
  th.mark = '{}({})'.format(model_name, th.data_config.split(':')[0])
  th.gather_summ_name = th.prefix + summ_name + '.sum'
  core.activate()


if __name__ == '__main__':
  console.suppress_logging()
  tf.app.run()

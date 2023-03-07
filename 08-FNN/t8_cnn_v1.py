import fnn_core as core
import fnn_mu as m

from tframe import console
from tframe import tf



# -----------------------------------------------------------------------------
# Define model here
# -----------------------------------------------------------------------------
model_name = 'cnn_v1'
id = 1
def model():
  th = core.th
  model = m.get_initial_model()

  for layer in th.archi_string.split('-'):
    if layer[0] == 's': stride, c = 2, int(layer[1:])
    else: stride, c = 1, int(layer)

    model.add(m.mu.HyperConv1D(
      c, th.kernel_size, stride, activation='relu',
      use_batchnorm=th.use_batchnorm))

  return m.finalize(model, flatten=True)


def main(_):
  console.start('{} on FNN-SSC task'.format(model_name.upper()))

  th = core.th
  th.rehearse = 0
  # ---------------------------------------------------------------------------
  # 0. date set setup
  # ---------------------------------------------------------------------------
  th.data_config = 'sleepedfx 1,2'
  th.input_shape = [3000, 2]

  # ---------------------------------------------------------------------------
  # 1. folder/file names and device
  # ---------------------------------------------------------------------------
  th.update_job_dir(id, model_name)
  th.set_date_as_prefix()
  summ_name = model_name

  th.visible_gpu_id = 0
  # ---------------------------------------------------------------------------
  # 2. model setup
  # ---------------------------------------------------------------------------
  th.model = model

  th.kernel_size = 3
  th.activation = 'relu'

  th.archi_string = '128-s128-s128-s64-s32'
  # ---------------------------------------------------------------------------
  # 3. trainer setup
  # ---------------------------------------------------------------------------
  th.epoch = 1000
  th.early_stop = True

  th.batch_size = 32

  th.optimizer = 'adam'
  th.learning_rate = 0.0003
  th.balance_classes = True

  th.train = True
  th.overwrite = True
  # ---------------------------------------------------------------------------
  # 4. other stuff and activate
  # ---------------------------------------------------------------------------
  th.mark = '{}({})'.format(model_name, th.archi_string)
  th.gather_summ_name = th.prefix + summ_name + '.sum'
  core.activate()



if __name__ == '__main__':
  console.suppress_logging()
  tf.app.run()

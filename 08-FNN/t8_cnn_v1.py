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
    if layer[0] == 's': stride, c = 3, int(layer[1:])
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
  th.data_config += ' val_ids=16,17 test_ids=18,19'
  # th.data_config += ' preprocess=iqr'
  th.data_config += ' sg_preprocess=trim;iqr'
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
  th.use_batchnorm = True

  th.archi_string = '16-s16-32-s32-64'
  # ---------------------------------------------------------------------------
  # 3. trainer setup
  # ---------------------------------------------------------------------------
  th.epoch = 20000
  th.early_stop = True
  th.batch_size = 64

  th.optimizer = 'adam'
  th.learning_rate = 0.0001
  th.balance_classes = True

  th.train = True
  th.patience = 20
  th.overwrite = True

  th.validate_train_set = True
  th.epoch_delta = 0.1
  # ---------------------------------------------------------------------------
  # 4. other stuff and activate
  # ---------------------------------------------------------------------------
  th.mark = '{}({})'.format(model_name, th.archi_string)
  th.gather_summ_name = th.prefix + summ_name + '.sum'
  core.activate()



if __name__ == '__main__':
  console.suppress_logging()
  tf.app.run()

import s2s_core as core
import s2s_mu as m

from tframe import console
from tframe import tf



# -----------------------------------------------------------------------------
# Define model here
# -----------------------------------------------------------------------------
model_name = 'cnn_bd'
id = 2
def model():
  from tframe.layers.common import BatchReshape

  th = core.th
  model = m.get_initial_model()

  for layer in th.archi_string.split('-'):
    if layer[0] == 's': stride, c = 2, int(layer[1:])
    else: stride, c = 1, int(layer)

    model.add(m.mu.HyperConv1D(
      c, th.kernel_size, stride, activation='relu',
      use_batchnorm=th.use_batchnorm))

  model.add(BatchReshape())

  return m.finalize(model, flatten=True, use_gap=True)


def main(_):
  console.start('{} on S2S-SSC task'.format(model_name.upper()))

  th = core.th
  th.rehearse = 0
  # ---------------------------------------------------------------------------
  # 0. date set setup
  # ---------------------------------------------------------------------------
  th.data_config = 'sleepeason1 1,2'
  th.data_config += ' val_ids=16,17 test_ids=18,19'

  th.epoch_num = 5
  th.eval_epoch_num = 10
  th.sg_buffer_size = 10
  # th.input_shape = [3000 * th.epoch_num, len(th.fusion_channels[0])]
  th.input_shape = [None, len(th.fusion_channels[0])]
  th.use_batch_mask = True

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

  th.archi_string = '16-s16-32-s32-64-5'
  # ---------------------------------------------------------------------------
  # 3. trainer setup
  # ---------------------------------------------------------------------------
  th.epoch = 10000
  th.early_stop = True
  th.batch_size = 64

  th.optimizer = 'adam'
  th.learning_rate = 0.0001
  th.balance_classes = True

  th.train = True
  th.patience = 5
  th.overwrite = True

  th.updates_per_round = 50
  # ---------------------------------------------------------------------------
  # 4. other stuff and activate
  # ---------------------------------------------------------------------------
  th.mark = '{}({})'.format(model_name, th.archi_string)
  th.gather_summ_name = th.prefix + summ_name + '.sum'
  core.activate()



if __name__ == '__main__':
  console.suppress_logging()
  tf.app.run()

import attn_core as core
import attn_mu as m

from tframe import console
from tframe import tf



# -----------------------------------------------------------------------------
# Define model here
# -----------------------------------------------------------------------------
model_name = 'attn_pad2'
id = 6
def model():
  # from tframe.layers.common import BatchReshape

  th = core.th
  model = m.get_initial_model()

  m.add_deep_sleep_net_lite(model, 64)
  m.add_AFR(model)
  m.add_EncodeLayer(model)
  m.add_EncodeLayer(model)

  # model.add(BatchReshape())

  return m.finalize(model, flatten=True, use_gap=False)


def main(_):
  console.start('{} on Attn_Sleep task'.format(model_name.upper()))

  th = core.th
  th.rehearse = 0
  # ---------------------------------------------------------------------------
  # 0. date set setup
  # ---------------------------------------------------------------------------
  th.data_config = 'sleepeasonx EEGx2,EOGx1 beta'
  th.data_config += ' pattern=.*(sleepedfx)'
  # th.data_config += ' pattern=.*(ucddb)'
  # th.data_config += ' pattern=.*(rrsh)'

  # th.pp_config = 'alpha-1:8'

  th.epoch_num = 1
  th.eval_epoch_num = 1
  th.sg_buffer_size = 15
  th.epoch_pad = 2

  # th.input_shape = [None, th.input_channels]
  if th.epoch_pad > 0:
    assert th.epoch_num == th.eval_epoch_num == 1
    L = 128 * 30 * (1 + 2 * th.epoch_pad)
  else: L = 128 * 30 * th.epoch_num
  th.input_shape = [L * th.epoch_num, th.input_channels]
  th.use_batch_mask = True

  # th.dtype = tf.float16
  # ---------------------------------------------------------------------------
  # 1. folder/file names and device
  # ---------------------------------------------------------------------------
  th.update_job_dir(id, model_name)
  th.set_date_as_prefix()
  summ_name = model_name

  th.visible_gpu_id = 0
  th.suffix = '_1'
  # ---------------------------------------------------------------------------
  # 2. model setup
  # ---------------------------------------------------------------------------
  th.model = model

  th.activation = 'relu'

  th.filters = 64
  th.dropout = 0.5
  th.use_batchnorm = True
  # ---------------------------------------------------------------------------
  # 3. trainer setup
  # ---------------------------------------------------------------------------
  th.epoch = 10000

  th.early_stop = True
  th.batch_size = 256

  # th.batchlet_size = 128
  # th.gradlet_in_device = 1

  th.optimizer = 'adam'
  th.learning_rate = 0.0006
  th.balance_classes = True
  th.epoch_delta = 0

  th.global_l2_penalty = 0.002

  th.train = True
  th.patience = 50
  th.overwrite = True

  th.updates_per_round = 50
  # ---------------------------------------------------------------------------
  # 4. other stuff and activate
  # ---------------------------------------------------------------------------
  th.mark = '{}({})'.format(model_name, th.filters)
  th.gather_summ_name = th.prefix + summ_name + '.sum'
  core.activate()



if __name__ == '__main__':
  console.suppress_logging()
  tf.app.run()

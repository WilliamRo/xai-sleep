import s2s_core as core
import s2s_mu as m

from tframe import console
from tframe import tf



# -----------------------------------------------------------------------------
# Define model here
# -----------------------------------------------------------------------------
model_name = 'dtp'
id = 4
def model():
  model = m.get_initial_model()

  m.add_densely_connected_temporal_pyramids(model)

  return m.finalize(model, use_gap=True)


def main(_):
  console.start('{} on S2S-SSC task'.format(model_name.upper()))

  th = core.th
  th.rehearse = 0
  # ---------------------------------------------------------------------------
  # 0. date set setup
  # ---------------------------------------------------------------------------
  th.data_config = 'sleepeason1 EEGx1,EOGx1 alpha'
  th.data_config += ' pattern=.*(sleepedfx)'
  # th.data_config += ' pattern=.*(ucddb)'
  # th.data_config += ' pattern=.*(rrsh)'

  th.epoch_num = 1
  th.eval_epoch_num = 1
  th.sg_buffer_size = 10
  th.epoch_pad = 0

  # th.input_shape = [None, th.input_channels]
  if th.epoch_pad > 0:
    assert th.epoch_num == th.eval_epoch_num == 1
    L = 128 * 30 * (1 + 2 * th.epoch_pad)
  else: L = 128 * 30 * th.epoch_num
  th.input_shape = [L, th.input_channels]
  th.use_batch_mask = False
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

  th.activation = 'lrelu'
  th.use_batchnorm = False

  th.dtp_en_filters = 64
  th.dtp_en_ks = 128 // 2
  th.filters = 128
  th.kernel_size = 9

  th.dtpM = 3
  th.dtpR = 2
  # ---------------------------------------------------------------------------
  # 3. trainer setup
  # ---------------------------------------------------------------------------
  th.epoch = 10000

  th.early_stop = True
  th.batch_size = 64

  th.optimizer = 'adam'
  th.learning_rate = 0.001
  th.balance_classes = True
  th.epoch_delta = 0.1

  th.global_l2_penalty = 0.003

  th.train = True
  th.patience = 40
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

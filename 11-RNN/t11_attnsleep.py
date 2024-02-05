import rnn_core as core
import rnn_mu as m

from tframe import console
from tframe import tf



# -----------------------------------------------------------------------------
# Define model here
# -----------------------------------------------------------------------------
model_name = 'attnsleep'
id = 5
def model():
  from freud.talos_utils.sleep_models.attnsleep import AttnSleep

  th = core.th
  model = m.get_initial_model()

  # Add dsn backbone
  AttnSleep(128, th.filters).add_to(model)

  model.add(m.mu.Flatten())

  if th.use_rnn: model.add(m.mu.GRU(state_size=th.state_size))
  else: model.add(m.mu.Dense(th.state_size, activation='tanh'))

  return m.finalize(model)


def main(_):
  console.start('{} on RNN-SSC task'.format(model_name.upper()))

  th = core.th
  th.rehearse = 1
  # ---------------------------------------------------------------------------
  # 0. date set setup
  # ---------------------------------------------------------------------------
  assert th.epoch_pad == 0

  th.data_config = 'sleepeasonx EEGx2,EOGx1 beta'

  th.input_shape = [128 * 30, th.input_channels]
  th.input_shape = [100 * 30, th.input_channels]
  th.use_batch_mask = 1
  th.val_num_steps = 20
  th.eval_num_steps = 20
  th.val_batch_size = 1

  th.sg_buffer_size = 20
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
  th.use_rnn = 0

  th.filters = 64
  th.kernel_size = 3
  th.state_size = 64
  th.activation = 'relu'
  th.dropout = 0.5
  # ---------------------------------------------------------------------------
  # 3. trainer setup
  # ---------------------------------------------------------------------------
  th.class_weights = [0.4, 1.2, 0.3, 2.0, 1.0]
  th.loss_string = 'wce'
  # th.loss_string = 'cross_entropy'

  th.epoch = 10000
  th.early_stop = True
  th.batch_size = 32

  epoch_num = 400
  th.num_steps = 20
  if th.use_rnn: th.epoch_num = epoch_num
  else: th.epoch_num = 1

  th.optimizer = 'adam'
  th.learning_rate = 0.000

  th.train = 1
  th.patience = 20
  th.overwrite = 1

  vpr = 5
  if th.use_rnn:
    th.validate_cycle = epoch_num // th.num_steps * vpr
    th.val_batch_size = -1
    th.eval_batch_size = 1
  else:
    th.updates_per_round = epoch_num // th.num_steps
    th.validation_per_round = 1 / vpr
    th.val_batch_size = 256
    th.eval_batch_size = 256
    th.batch_size = th.batch_size * th.num_steps
  # ---------------------------------------------------------------------------
  # 4. other stuff and activate
  # ---------------------------------------------------------------------------
  th.mark = '{}(dsn{}-s{})'.format(model_name, th.filters, th.state_size)
  if th.use_rnn: th.mark += '-R'
  th.gather_summ_name = th.prefix + summ_name + '.sum'
  core.activate()



if __name__ == '__main__':
  console.suppress_logging()
  tf.app.run()

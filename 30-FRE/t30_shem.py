import fre_core as core
import fre_mu as m

from tframe import console
from tframe import tf



# -----------------------------------------------------------------------------
# Define model here
# -----------------------------------------------------------------------------
model_name = 'shem'
id = 1
def model():
  from tframe.layers.common import BatchReshape

  th = core.th
  model = m.get_initial_model()

  m.add_deep_sleep_net_lite(model, th.filters)

  # model.add(BatchReshape())

  return m.finalize(model, flatten=True, use_gap=False)



def main(_):
  console.start('{} on FRE-SSC task'.format(model_name.upper()))

  th = core.th
  th.rehearse = 0
  # ---------------------------------------------------------------------------
  # 0. date set setup
  # ---------------------------------------------------------------------------
  th.data_config = 'sleepeasonx EEGx2 beta'
  th.tgt_config = 'W:0;N12R:1,2,4;N3:3'
  # th.tgt_config = 'Wake:0;N1:1;N2:2;N3:3;RXXX:4'

  th.num_classes = len(th.tgt_tuples)

  th.epoch_num = 1
  th.eval_epoch_num = 1
  th.sg_buffer_size = 15

  L = 128 * 30 * th.epoch_num
  th.input_shape = [L, th.input_channels]

  th.use_batch_mask = 0
  # ---------------------------------------------------------------------------
  # 1. folder/file names and device
  # ---------------------------------------------------------------------------
  th.update_job_dir(id, model_name)
  th.set_date_as_prefix()
  summ_name = model_name

  th.visible_gpu_id = 0
  # th.suffix = ''
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
  th.epoch = 100

  th.early_stop = True
  th.batch_size = 128

  th.optimizer = 'adam'
  th.learning_rate = 0.0003
  th.balance_classes = True
  th.epoch_delta = 0.1

  # th.global_l2_penalty = 0.002

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

import pyco_core as core
import pyco_mu as m

from tframe import console
from tframe import tf

from freud.talos_utils.slp_agent import SleepAgent, SleepEason


# -----------------------------------------------------------------------------
# Define model here
# -----------------------------------------------------------------------------
model_name = 'sleepyco'
id = 1
def model():

  from pyco_layers.utils import BatchReshape
  from pyco_layers.classifiers import Transformer_Encoder, Attention
  th = core.th

  model = m.get_initial_model()

  # sleepyco backbone
  model.add(BatchReshape())

  # init_layer
  m.make_layers(model, 1, 64, 2, None,
                'init', True)
  # layer_1
  m.make_layers(model, 64, 128, 2, 5,
                '1')
  # layer_2
  m.make_layers(model, 128, 192, 3, 5,
                '2')
  # layer_3
  m.make_layers(model, 192, 256, 3, 5,
                '3')
  # layer_4
  m.make_layers(model, 256, 256, 3, 5,
                '4')
  # if single
  m.get_features(model)

  model.add(BatchReshape(reverse=True))

  seq_encoder = Transformer_Encoder(128, 1024, 6,
                                    8, th.epoch_num, 0.1,
                                    0.1)


  model.add(seq_encoder)

  model.add(Attention(64))






  return m.finalize(model, flatten=True, use_gap=False)


def main(_):
  console.start('{} on S2S-SSC task'.format(model_name.upper()))

  th = core.th
  th.rehearse = 0
  # ---------------------------------------------------------------------------
  # 0. date set setup
  # ---------------------------------------------------------------------------
  # th.data_config = 'sleepeasonx EEGx2,EOGx1 beta'
  SleepAgent.register_dataset("sleepeasonzscore", SleepEason)

  th.data_config = 'sleepeasonzscore EEGx1 beta'
  th.data_config += ' pattern=.*(sleepedfx)'
  # th.data_config += ' pattern=.*(ucddb)'
  # th.data_config += ' pattern=.*(rrsh)'

  # th.pp_config = 'alpha-1:8'

  th.epoch_num = 64
  th.eval_epoch_num = 64
  th.sg_buffer_size = 20

  # th.input_shape = [None, th.input_channels]
  if th.epoch_pad > 0:
    assert th.epoch_num == th.eval_epoch_num == 1
    L = 100 * 30 * (1 + 2 * th.epoch_pad)
  else: L = 100 * 30 * th.epoch_num
  th.input_shape = [L, th.input_channels]
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
  th.batch_size = 1

  # th.batchlet_size = 128
  # th.gradlet_in_device = 1

  th.optimizer = 'adam'
  th.learning_rate = 0.0006
  th.balance_classes = True

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

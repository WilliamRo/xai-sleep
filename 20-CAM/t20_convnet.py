import cam_core as core
import cam_mu as m

from tframe import console
from tframe import tf



# -----------------------------------------------------------------------------
# Define model here
# -----------------------------------------------------------------------------
model_name = 'cam'
id = 1
def model(): return m.get_data_fusion_model()


def main(_):
  console.start('{} on FNN-SSC task'.format(model_name.upper()))

  th = core.th
  th.rehearse = False
  # ---------------------------------------------------------------------------
  # 0. date set setup
  # ---------------------------------------------------------------------------
  th.data_config = 'sleepedfx 1,2'
  th.data_config += ' val_ids=16,17 test_ids=18,19'
  # th.data_config += ' preprocess=iqr'
  th.data_config += ' sg_preprocess=trim;iqr'
  th.input_shape = [3000, len(th.fusion_channels[0])]

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

  th.archi_string = '48-48-m-32-32-m-16'
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

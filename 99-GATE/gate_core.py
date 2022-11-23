import sys, os

#: Add necessary paths to system path list so that all task modules with
#:  filename `tXX_YYY.py` can be run directly.
#:
#: Recommended project structure:
#: DEPTH  0          1         2 (*)
#:        this_proj
#:                |- 01-MNIST
#:                          |- mn_core.py
#:                          |- mn_du.py
#:                          |- mn_mu.py
#:                          |- t1_lenet.py
#:                |- 02-CIFAR10
#:                |- ...
#:                |- tframe
#:
# ! Specify the directory depth with respect to the root of your project here
DIR_DEPTH = 2
ROOT = os.path.abspath(__file__)
for _ in range(DIR_DEPTH):
  ROOT = os.path.dirname(ROOT)
  if sys.path[0] != ROOT: sys.path.insert(0, ROOT)
# =============================================================================
from tframe import console
from xsleep.slp_config import SLPConfig as Hub
# from tframe import DefaultHub as Hub
from tframe import Classifier

import gate_du as du

# -----------------------------------------------------------------------------
# Initialize config and set data/job dir
# -----------------------------------------------------------------------------
th = Hub(as_global=True)
th.config_dir(dir_depth=1)
job_dir = th.job_dir

# -----------------------------------------------------------------------------
# Device configuration
# -----------------------------------------------------------------------------
th.allow_growth = False
th.gpu_memory_fraction = 0.50

# -----------------------------------------------------------------------------
# Set common trainer configs
# -----------------------------------------------------------------------------
# th.input_shape = [120, 2]
th.input_shape = [3000, 3]

th.early_stop = True
th.patience = 20

th.print_cycle = 10
th.validation_per_round = 2

th.export_tensors_upon_validation = True

th.evaluate_train_set = True
th.evaluate_val_set = True
th.evaluate_test_set = True

th.val_batch_size = 100
th.eval_batch_size = 100

# per sample length
th.random_sample_length = 3000


def activate():
  # Build model
  assert callable(th.model)
  model = th.model()
  assert isinstance(model, Classifier)

  # Rehearse if required
  if th.rehearse:
    model.rehearse(export_graph=True, build_model=False,
                   path=model.agent.ckpt_dir, mark='model')
    return

  if not th.train:
    # Evaluate on test set
    import pickle
    dataset_name, data_num, _ = th.data_config.split(':')
    person_num = '(all)' if data_num == '' else f'({data_num})'
    prefix = dataset_name + person_num
    model_architecture = 'fnn'
    if th.use_rnn:
      model_architecture = 'rnn'
    tfd_format_path = os.path.join(th.data_dir, dataset_name,
                                   f'{prefix}-format-{model_architecture}{th.input_shape[0]}.tfds')
    if os.path.exists(tfd_format_path):
      with open(tfd_format_path, 'rb') as _input_:
        console.show_status(f'loading {tfd_format_path}...')
        dataset = pickle.load(_input_)
        du.SLPAgent.evaluate_model(model, dataset)

  else:
    # Load data
    train_set, val_set, test_set = du.load_data()
    if th.centralize_data: th.data_mean = train_set.feature_mean

    model.train(train_set, validation_set=val_set, test_set=test_set,
                trainer_hub=th)

    model = model.agent.launch_model

  # End
  model.shutdown()
  console.end()

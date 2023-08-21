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
#! Specify the directory depth with respect to the root of your project here
DIR_DEPTH = 2
ROOT = os.path.abspath(__file__)
for _ in range(DIR_DEPTH):
  ROOT = os.path.dirname(ROOT)
  if sys.path[0] != ROOT: sys.path.insert(0, ROOT)
# =============================================================================
from tframe import console
from tframe import Classifier

from freud.talos_utils.slp_config import SleepConfig as Hub

import s2s_du as du


# -----------------------------------------------------------------------------
# Initialize config and set data/job dir
# -----------------------------------------------------------------------------
th = Hub(as_global=True)
th.config_dir()
th.data_dir = os.path.join(ROOT, 'data')

# -----------------------------------------------------------------------------
# Device configuration
# -----------------------------------------------------------------------------
th.allow_growth = False
th.gpu_memory_fraction = 0.5

# -----------------------------------------------------------------------------
# Data configuration
# -----------------------------------------------------------------------------
th.input_shape = None  # TODO
th.num_classes = 5
th.output_dim = th.num_classes
th.balance_training_stages = True

# -----------------------------------------------------------------------------
# Set common trainer configs
# -----------------------------------------------------------------------------
th.early_stop = True
th.patience = 5

th.print_cycle = 2
th.updates_per_round = 50
th.validation_per_round = 1

th.export_tensors_upon_validation = True

th.val_batch_size = 64
th.val_progress_bar = True



def activate():
  if 'deactivate' in th.developer_code: return

  # Build model
  assert callable(th.model)
  model = th.model()
  assert isinstance(model, Classifier)

  if th.rehearse:
    model.rehearse(build_model=False)
    return

  # Load data
  train_set, val_set, test_set = du.load_data()

  # Train or evaluate
  if th.train:
    model.train(train_set, validation_set=val_set,
                test_set=test_set, trainer_hub=th)
  else:
    pass

  # Evaluate best model
  model.agent.load()
  for ds in (train_set.validation_set, val_set, test_set):
    model.evaluate_pro(ds, batch_size=128, verbose=True,
                       show_confusion_matrix=True,
                       plot_confusion_matrix=False,
                       show_class_detail=True)

  # End
  model.shutdown()
  console.end()

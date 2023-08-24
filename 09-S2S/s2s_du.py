from freud.talos_utils.slp_agent import SleepAgent
from freud.talos_utils.slp_set import DataSet

from tframe.trainers.trainer import Trainer
from tframe.models.sl.classifier import ConfusionMatrix



def load_data():
  from s2s_core import th

  data_sets = SleepAgent.load_data()

  js = [1, 2]
  # if th.epoch_delta == 0: js.append(0)
  for j in js: data_sets[j] = data_sets[j].validation_set

  return data_sets


def evaluate(trainer: Trainer):
  from s2s_core import th
  from tframe import Classifier

  model: Classifier = trainer.model
  agent = model.agent

  # Evaluate val_set and test_set
  evaluate_pro = lambda ds: model.evaluate_pro(
    ds, batch_size=th.val_batch_size, show_confusion_matrix=True,
    show_class_detail=True)

  train_set, val_set, test_set = (
    trainer.training_set, trainer.validation_set, trainer.test_set)
  train_set = train_set.validation_set

  train_cm: ConfusionMatrix = evaluate_pro(train_set)
  val_cm: ConfusionMatrix = evaluate_pro(val_set)
  test_cm: ConfusionMatrix = evaluate_pro(test_set)

  # Take down results to note
  agent.put_down_criterion('Train Accuracy', train_cm.accuracy)
  agent.put_down_criterion('Train F1', train_cm.macro_F1)
  agent.put_down_criterion('Test Accuracy', test_cm.accuracy)
  agent.put_down_criterion('Test F1', test_cm.macro_F1)

  for ds, cm in zip((train_set, val_set, test_set),
                    (train_cm, val_cm, test_cm)):
    agent.take_notes(f'Results of `{ds.name}` dataset:')
    agent.take_notes(str(cm.matrix_table()), prompt='')
    agent.take_notes(str(cm.make_table(decimal=4, class_details=True)),
                     prompt='')



if __name__ == '__main__':
  from fnn_core import th

  th.data_config = 'sleepedfx 1,2'
  th.data_config += ' val_ids=12,13,14,15 test_ids=16,17,18,19'
  # th.data_config += ' preprocess=iqr mad=10'

  train_set, _, _ = load_data()

  assert isinstance(train_set, DataSet)

  for batch in train_set.gen_batches(50, is_training=True):
    print()





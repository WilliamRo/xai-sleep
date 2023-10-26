from freud.talos_utils.slp_agent import SleepAgent
from freud.talos_utils.slp_set import DataSet

from tframe.trainers.trainer import Trainer
from tframe.models.sl.classifier import ConfusionMatrix



def load_data():
  from fre_core import th

  data_sets = SleepAgent.load_data()

  js = [1, 2]
  # if th.epoch_delta == 0: js.append(0)
  for j in js: data_sets[j] = data_sets[j].validation_set

  return data_sets



def evaluate(trainer: Trainer):
  from fre_core import th
  from tframe import Classifier

  model: Classifier = trainer.model
  agent = model.agent

  # Evaluate val_set and test_set
  evaluate_pro = lambda ds: model.evaluate_pro(
    ds, batch_size=th.val_batch_size, show_confusion_matrix=True,
    show_class_detail=True)

  datasets = [trainer.validation_set, trainer.test_set]
  if th.evaluate_train_set:
    datasets.insert(0, trainer.training_set.validation_set)

  cms = [evaluate_pro(ds) for ds in datasets]

  # Take down results to note
  if th.evaluate_train_set:
    agent.put_down_criterion('Train Accuracy', cms[0].accuracy)
    agent.put_down_criterion('Train F1', cms[0].macro_F1)

  agent.put_down_criterion('Test Accuracy', cms[-1].accuracy)
  agent.put_down_criterion('Test F1', cms[-1].macro_F1)

  for ds, cm in zip(datasets, cms):
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





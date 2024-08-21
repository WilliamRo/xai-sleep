from pictor.xomics.ml.ml_engine import MLEngine
from tframe.utils.sklearn_api.base_estimator import BaseEstimator
from tframe import tf, hub as th

import numpy as np



class SKENet(BaseEstimator):

  def __init__(self, lr=0.01, patience=10, max_iterations=1e9,
               alpha=1.0, l1_ratio=0.5, **hp):
    super().__init__(lr, patience, max_iterations, **hp)
    self.alpha = alpha
    self.l1_ratio = l1_ratio


  def _init_np_variables(self, X: np.ndarray):
    self._np_variables['W'] = np.zeros(shape=[X.shape[1], 1])
    self._np_variables['b'] = np.zeros(shape=[1])


  def _predict_np(self, X):
    # X.shape = [?, D]
    W, b = self._np_variables['W'], self._np_variables['b']
    y = np.matmul(X, W) + b
    return y


  def _predict_tf(self, X):
    # X.shape = [?, D]
    W, b = self.tf_var_dict['W'], self.tf_var_dict['b']
    return tf.add(tf.matmul(X, W), b)


  def _loss_function_tf(self, y_true, y_pred):
    """ 1 / (2 * n_samples) * ||y - Xw||^2_2
        + alpha * l1_ratio * ||w||_1
        + 0.5 * alpha * (1 - l1_ratio) * ||w||^2_2 """
    W = self.tf_var_dict['W']
    l0 = 0.5 * tf.reduce_mean(tf.square(y_true - y_pred))
    l1 = self.alpha * self.l1_ratio * tf.reduce_sum(tf.abs(W))
    l2 = 0.5 * self.alpha * (1 - self.l1_ratio) * tf.reduce_sum(tf.square(W))

    loss = tf.add_n([l0, l1, l2])
    return loss



class TFENet(MLEngine):

  abbreviation = 'tfENET'

  SK_CLASS = SKENet
  IS_CLASSIFIER = False

  DEFAULT_HP_SPACE = {
    # 'alpha': np.logspace(-6, 1, 8),
    # 'l1_ratio': np.linspace(0, 1, 5),
    'alpha': [0.5, 1.0],
    'l1_ratio': [0.0, 0.5, 1.0],
  }

  DISABLE_PARALLEL = True

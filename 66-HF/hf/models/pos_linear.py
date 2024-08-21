from pictor.xomics.ml.ml_engine import MLEngine
from sklearn.linear_model import LinearRegression as SKLinearRegression
from tframe.utils.sklearn_api.base_estimator import BaseEstimator
from tframe import tf

from sklearn.linear_model import ElasticNet as SKElasticNet
from sklearn import svm

import numpy as np



class SKSoftLinear(BaseEstimator):

  def __init__(self, lr=0.01, patience=2, max_iterations=1e9,
               alpha=1.0, **hp):
    super().__init__(lr, patience, max_iterations, **hp)
    self.alpha = alpha


  def _init_np_variables(self, X: np.ndarray):
    self._np_variables['W'] = np.zeros(shape=[X.shape[1], 1])
    self._np_variables['b'] = np.zeros(shape=[1])


  def _predict_np(self, X):
    # X.shape = [?, D]
    W, b = self._np_variables['W'], self._np_variables['b']
    y = np.log(1. + np.exp(np.matmul(X, W) + b))
    return y


  def _predict_tf(self, X):
    # X.shape = [?, D]
    W, b = self.tf_var_dict['W'], self.tf_var_dict['b']
    y = tf.log(1. + tf.exp(tf.matmul(self.tf_X, W) + b))
    return y


  def _loss_function_tf(self, y_true, y_pred):
    delta = y_true - y_pred
    l1 = tf.reduce_mean(tf.square(delta))

    y_true_mu = tf.reduce_mean(y_true)
    delta_mu = tf.reduce_mean(delta)
    l2 = tf.reduce_mean(tf.abs((y_true_mu - y_true) * (delta_mu - delta)))
    return l1 + self.alpha * l2



class SoftLinear(MLEngine):

  abbreviation = 'SLiR'

  SK_CLASS = SKSoftLinear
  IS_CLASSIFIER = False

  DEFAULT_HP_SPACE = {
    # 'lr': np.logspace(-5, -1, 5),
    # 'alpha': np.logspace(-4, 0, 5),
    'alpha': [0.0, 0.5, 1.0],
  }

  DISABLE_PARALLEL = True

import numpy as np
import pprint

from roma import Nomear
from tframe import console
from tframe import tf

console.suppress_logging()



class MatOptimizer(Nomear):

  def __init__(self, F1, F2, lr=0.001, top_K=1):

    self.np_F1 = F1
    self.np_F2 = F2
    self.N = F1.shape[0]
    console.show_status(f'D = {self.feature_dim}')
    self.top_K = top_K

    self.F1 = tf.placeholder(tf.float32, [self.N, self.feature_dim], 'X1')
    self.F2 = tf.placeholder(tf.float32, [self.N, self.feature_dim], 'X2')
    self.w = tf.get_variable('w', [1, self.feature_dim], tf.float32,
                             initializer=tf.zeros_initializer())
    self.w = tf.nn.sigmoid(self.w)

    self.X1, self.X2 = self.F1 * self.w, self.F2 * self.w

    self.dist_mat = None
    self.define_dist_mat()

    self.session = tf.Session()
    self.session.run(tf.global_variables_initializer())

    self.loss = None
    self.metric = None
    self.lr = lr
    self.train_step = None

    self.define_loss()
    self.define_train_step()


  @property
  def feature_dim(self): return self.np_F1.shape[1]

  @property
  def feed_dict(self): return {self.F1: self.np_F1, self.F2: self.np_F2}

  def calc_dist_mat(self):
    return self.session.run(self.dist_mat, feed_dict=self.feed_dict)

  def define_dist_mat(self, version='prod'):
    if version == 'prod':
      self.X1 = self.X1 / tf.norm(self.X1, axis=1, keepdims=True)
      self.X2 = self.X2 / tf.norm(self.X2, axis=1, keepdims=True)
      self.dist_mat = 1. - tf.matmul(self.X1, self.X2, transpose_b=True)
    elif version == 'norm':
      # self.X1.shape = (N, D)
      X1, X2 = tf.expand_dims(self.X1, 1), tf.expand_dims(self.X2, 0)
      X1 = tf.broadcast_to(X1, [self.N, self.N, self.feature_dim])
      X2 = tf.broadcast_to(X2, [self.N, self.N, self.feature_dim])
      self.dist_mat = tf.norm(X1 - X2, axis=2)

  def define_loss(self, version='simple', lamb_sigma=2):
    if version == 'top':
      diag = tf.diag_part(self.dist_mat)
      loss_diag = tf.reduce_mean(diag)

      safe_rate = 0.3
      k = int((1 - safe_rate) * self.N)
      top_K, ind = tf.nn.top_k(self.dist_mat, k=k, sorted=True)
      loss_top_K = tf.reduce_mean(top_K)

      lambd = 10
      self.loss = lambd * loss_diag - loss_top_K
    elif version == 'simple':
      a = -np.ones(shape=(self.N, self.N), dtype=float) / self.N
      np.fill_diagonal(a, self.N)

      a = tf.constant(a, dtype=tf.float32)
      self.loss = tf.reduce_mean(tf.square(a * self.dist_mat))
    elif version == 'max':
      indices = tf.argmin(self.dist_mat, axis=1)
      correct = tf.equal(indices, tf.range(self.N, dtype=tf.int64))
      signs = 1. - tf.cast(correct, tf.float32)
      highest = tf.reduce_min(self.dist_mat, axis=1)
      self.loss = tf.reduce_mean(signs * highest)
    elif version == 'entropy':
      y = np.zeros(shape=(self.N, self.N), dtype=float)
      np.fill_diagonal(y, 1)
      y = tf.constant(y, dtype=tf.float32)

      self.loss = tf.losses.softmax_cross_entropy(
        onehot_labels=y, logits=-self.dist_mat)
    else: raise NotImplementedError

    # Add sigma if necessary
    if lamb_sigma > 0:
      sigma = tf.math.reduce_std(self.dist_mat)
      self.loss = self.loss - lamb_sigma * sigma

  def define_train_step(self):
    op = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
    self.train_step = op.minimize(self.loss)

  def take_a_train_step(self):
    fetches = [self.train_step, self.loss, self.dist_mat]
    _, loss, mat = self.session.run(fetches, feed_dict=self.feed_dict)

    # Calculate accuracy
    top_K = self.top_K
    indices = np.argsort(mat, axis=1)[:, :top_K]
    match = np.arange(self.N).reshape([-1, 1]) == indices
    match = np.max(match, axis=1)
    acc = np.mean(match)

    return loss, acc, mat

  def fit(self, max_steps=1000, print_cycle=20, print_w=False):
    losses, accs, mats, steps = [], [], [], []
    for i in range(max_steps):
      loss, acc, mat = self.take_a_train_step()
      if i % print_cycle != 0: continue
      losses.append(loss)
      accs.append(acc)
      mats.append(mat)
      steps.append(i)
      console.show_status(f'[{i}] Loss: {loss:.4f} Acc: {acc:.4f}')
      if print_w:
        w = self.session.run(self.w)
        pprint.pprint(w)

    return losses, accs, mats, steps



if __name__ == '__main__':
  F1 = np.random.random((100, 30))
  F2 = np.random.random((100, 30))

  optimizer = MatOptimizer(F1, F2)

  loss_0 = optimizer.session.run(optimizer.loss, feed_dict=optimizer.feed_dict)
  print(loss_0)
  loss_1, acc = optimizer.take_a_train_step()
  loss_1, acc = optimizer.take_a_train_step()
  print(loss_1)

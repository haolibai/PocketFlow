import tensorflow as tf
from tensorflow import digamma as digamma
from tensorflow import lgamma as gammaln
import numpy as np
from utils import UniformQuantization
import os
import pdb

# NOTE: below are hypers for pruning

tf.app.flags.DEFINE_float("pi_zero", 0.95, "pi for zero componnet")
tf.app.flags.DEFINE_integer("T", 31, "maximum number of mixtures")
# flags.DEFINE_float("truncate", 0.98, "dpmm truncate value")
tf.app.flags.DEFINE_float("alpha_0", 10., "prior for concentration")
tf.app.flags.DEFINE_float("lr_weight", 5e-4, "learning rate for fine tuning weights")
tf.app.flags.DEFINE_float("degree_0", 20, "prior for degree_0, usually equals to T")
tf.app.flags.DEFINE_bool("sample", False, "use weight sampling to speed up dpmm")
tf.app.flags.DEFINE_float("sample_ratio", 0.1, "the sample size of weights for dpmm")
tf.app.flags.DEFINE_bool("uniform", False, "uniform quantization, (no pruning should be assigned)")
tf.app.flags.DEFINE_bool("layerwise", False, "layerwise dpmm for clustering")
tf.app.flags.DEFINE_bool("bitwise", True, "pick bitwise number of components")
tf.app.flags.DEFINE_float("init_precs", 1., "initial value for precision variables")
tf.app.flags.DEFINE_float("dpmm_lr", 1e-2, 'dpmm finetune lr')

FLAGS = tf.app.flags.FLAGS

#TODO: tmp variables for testing. to be put in FLAGS
W_BITS = 4
T = 2 ** W_BITS
CUM_RATIO = 0.9
USE_BUCKET = False


def flatten(weights):
  assert len(weights.shape) == 4, 'Weights for quantization must be 4-D!'
  fw = tf.reshape(weights, [-1])
  num_weights = fw.get_shape()[0].value
  return fw, num_weights


def search_uniform_top(pis, cumulative_ratio=0.9):
  """ ALgorithms to uniformly pick out the quantization points
  whose cumulative r is > cumulative_ratio.
  As long as the first solution is found, stop and return the result
  The search begins from low bit to high bit
  """

  K = len(pis)
  Bit = int(np.log2(K))
  results_failed = []
  for b in range(1, Bit+1):
    results = {}

    if b == 1:
      # 1-bit: free combination
      for i in range(K):
        for j in range(i+1, K):
          results[(i, j)] = pis[i]+pis[j]
          if results[(i, j)] > cumulative_ratio:
            return (i, j), 1
    else:
      # for bits > 1, follow the algo
      k = 2 ** b
      imax = int(np.ceil((K-1)/k)) # interval_max
      for i in range(1, imax+1):
        # i: interval
        nums = int(K - (1+i*(k-1)) + 1)
        for n in range(nums):
          index = tuple([n+i*count for count in range(k)])
          results[index] = np.sum(pis[list(index)])
          if results[index] > cumulative_ratio:
            return index, b

    results_failed.append(results)


def neg_normal_log_likelihood(x, mean, prec, pi):
  diff = (tf.expand_dims(x, 1) - tf.expand_dims(mean, 0)) ** 2
  diff = -0.5 * prec * diff
  # apply the log sum exp trick
  tmp_max = tf.reduce_max(diff, 1)
  res = tf.reduce_sum(tf.sqrt(prec / (2. * np.pi)) * pi * tf.exp(diff - tf.expand_dims(tmp_max, 1)), 1)
  res = tf.log(res) + tmp_max
  return -res


class DPMixture(object):
  def __init__(self, weights, sess, uniform=True):
    """
    Args:
    * weights: a list of tensors, weights to be quantized
    """

    self.weights = weights
    self.weights_name = weights.name[:-2]
    self.sess = sess
    self.uniform = uniform
    self.flatten_weights, self.num_weights= flatten(self.weights)
    self.__build_dpmm()
    self.__build_top_mixtures()

    self.dpmm_opt = self.__dpmm_update_ops()
    self.proximal_op = self.__weights_updatel_ops()
    self.sess.run(self.init_op)

  def __build_dpmm(self):
    """ Build the variables of dpmm model """

    weights_num = self.flatten_weights.get_shape()[0].value

    if FLAGS.sample:
      # sample a minibatch of weights for update of DPMM variables

      # TODO: follow strictly the SVI method? a bit difficult to implement
      sample_index = np.arange(weights_num)
      np.random.shuffle(sample_index)
      self.num_sampled = int(FLAGS.sample_ratio*weights_num)
      sample_index = sample_index[:weights_num]
      sampled_weights = tf.gather(self.flatten_weights, sample_index)
    else:
      sampled_weights = self.flatten_weights

    # define quantization points, based on all weights
    w = UniformQuantization.clip_weights(sampled_weights)
    w_normalized, alpha, beta = UniformQuantization.scale(w, 'weight', USE_BUCKET)
    self.current_means = UniformQuantization.inv_scale(tf.linspace(0., 1., T), alpha, beta)
    self.means = tf.get_variable(os.path.join(self.weights_name, 'means'), dtype=tf.float32, \
        initializer=self.current_means, trainable=(not self.uniform))

    # a scalar. all clusters share the same precision variable
    self.precs = tf.get_variable(os.path.join(self.weights_name, 'precs'), dtype=tf.float32, \
        initializer=tf.constant(FLAGS.init_precs), trainable=False)

    # define the stick breaking atoms variational parameters
    init_log_gam_1 = np.log(np.tile([1. / T], T))
    init_log_gam_2 = np.log(np.tile([2.], [T]))
    self.log_gam_1 = tf.get_variable(os.path.join(self.weights_name, 'log_gam_1'), shape=[T], dtype=tf.float32,
                     initializer=tf.constant_initializer(init_log_gam_1))
    self.log_gam_2 = tf.get_variable(os.path.join(self.weights_name, 'log_gam_2'), shape=[T], dtype=tf.float32,
                     initializer=tf.constant_initializer(init_log_gam_2))

    self.gam_1 = tf.exp(self.log_gam_1)
    self.gam_2 = tf.exp(self.log_gam_2)

    # z: multinomial distributed; apply the log exp sum trick
    init_log_r = np.random.rand(weights_num, T)
    init_log_r = init_log_r / np.sum(init_log_r, 1)[:, np.newaxis]
    init_log_r = np.log(init_log_r)
    self.log_rho = tf.get_variable(os.path.join(self.weights_name, 'log_rho'), shape=[weights_num, T], \
        dtype=tf.float32, initializer=tf.constant_initializer(init_log_r))

    tmp_max = tf.tile(tf.reduce_max(self.log_rho, axis=1, keepdims=True), [1, T])
    self.r = tf.exp(self.log_rho - tmp_max) / tf.tile(
      tf.reduce_sum(tf.exp(self.log_rho - tmp_max), axis=1, keepdims=True), [1, T])

    # now the loss terms
    self.loss = - self.__get_elbo(sampled_weights)

  def __build_top_mixtures(self):
    self.expect_pis = (FLAGS.alpha_0 + tf.reduce_sum(self.r, axis=0)) / (T * FLAGS.alpha_0 + self.num_weights)
    self.top_ind = tf.placeholder(dtype=tf.int32, name='top_indices', shape=[None])
    self.top_means = tf.gather(self.means, self.top_ind)
    self.top_pis = tf.gather(self.expect_pis, self.top_ind)
    self.top_neg_likelihood = tf.reduce_mean(
      neg_normal_log_likelihood(self.flatten_weights, self.top_means, self.precs, pi=self.top_pis))

  def pick_top(self, mode='uniform'):
    """ picking up the top (and uniformly distributed) quantization points.
    Args:
    * mode: 'vanilla': directly truncate by FLAGS.ratio according to threshold and expect_pi
            'uniform' (default): uniformly truncate
            'cumulative_sum': cumulative sum of asending expect pis truncated by FLAGS.ratio
    Return:
    * top_ind: a numpy array of indices;
    """

    if mode in ['vanilla', 'cumulative_sum'] and self.uniform:
      raise ValueError("In uniform quantization, please use 'uniform' picking")

    expect_pis_numpy = self.sess.run(self.expect_pis)
    if mode == 'uniform':
      # apply the algo in numpy.
      top_ind, bits = search_uniform_top(expect_pis_numpy, cumulative_ratio=CUM_RATIO)
    else:
      raise ValueError("Wrong picking mode")

    return top_ind, bits

  def __dpmm_update_ops(self):
    # dpmm optimizer
    self.dpmm_step = tf.get_variable(name="dpmm_opt_step", initializer=tf.constant(1, dtype=tf.int32), trainable=False)
    self.update_mean_op = self.means.assign(self.current_means)
    self.update_prec_op = self.precs.assign(FLAGS.init_precs * tf.cast(self.dpmm_step, tf.float32))
    # TODO: how frequent to update means?
    with tf.control_dependencies([self.update_mean_op, self.update_prec_op]):
      optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.dpmm_lr).\
          minimize(self.loss, var_list=self.dpmm_trainable_vars, global_step=self.dpmm_step)
    return optimizer

  def __weights_updatel_ops(self):

    " proximal operator update of weights"
    # the quantized weight
    w_dims = self.weights.shape.__len__()
    num_quant_points = tf.cast(tf.shape(self.top_ind), tf.int64)
    shape_ = tf.concat((tf.ones(w_dims, dtype=tf.int64), num_quant_points), axis=0)
    w_new = tf.tile(tf.expand_dims(self.weights, w_dims), shape_)
    min_index = tf.argmin(tf.abs(w_new - self.top_means), axis=-1)

    # override gradient for the STE estimator (copied from non-uniform)
    self.quantized_weights = tf.gather(self.top_means, min_index)
    prox = (self.weights + 2*self.precs*self.quantized_weights)/(1 + 2*self.precs)
    return self.weights.assign(prox)

  @property
  def dpmm_trainable_vars(self):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='DPMM')

  @property
  def dpmm_global_vars(self):
    return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='DPMM')

  @property
  def init_op(self):
    return tf.variables_initializer(self.dpmm_global_vars)

  def __get_elbo(self, x):
    self.bound_v = self.__bound_v()
    self.bound_z = self.__bound_z()
    self.bound_likelihood = self.__bound_likelihood(x)
    return self.bound_v + self.bound_z + self.bound_likelihood

  def __bound_v(self):
    # todo: Different from PRML. This is from the paper and sklearn documentation. checked.
    bound = gammaln([1 + FLAGS.alpha_0]) - gammaln([FLAGS.alpha_0])
    bound = tf.tile(bound, [T])
    bound += (FLAGS.alpha_0 - 1) * (digamma(self.gam_2) - digamma(self.gam_1 + self.gam_2))
    bound += - gammaln(self.gam_1 + self.gam_2) + gammaln(self.gam_1) + gammaln(self.gam_2)
    bound += - (self.gam_1 - 1) * (digamma(self.gam_1) - digamma(self.gam_1 + self.gam_2))
    bound += - (self.gam_2 - 1) * (digamma(self.gam_2) - digamma(self.gam_1 + self.gam_2))
    return tf.reduce_sum(bound)

  def __bound_z(self):
    """
    Checked.
    """
    bound = 0
    for t in range(T):
      tmp = tf.reduce_sum(self.r[:, t + 1:] * (digamma(self.gam_2[t]) - digamma(self.gam_1[t] + self.gam_2[t])),
                axis=1)
      tmp += self.r[:, t] * (digamma(self.gam_1[t]) - digamma(self.gam_1[t] + self.gam_2[t]))
      tmp += - self.r[:, t] * tf.log(self.r[:, t] + np.finfo(np.float32).eps)
      bound += tf.reduce_sum(tmp)
    return bound

  def __bound_likelihood(self, x):
    """
    Output: Eq_logp_x: [N, T]
    Warnning: output is a vector. In ELBO, it needs to be summed over N.
    # checked consistent with prml solutions.
    # checked alter==Eq_logp_x is true
    optimizign this term is equivalent to proximal step
    """
    diff = tf.expand_dims(x, 1) - tf.expand_dims(self.means, 0)  # [N x T]
    diff = diff ** 2
    Eq_logp_x = self.r * (0.5 * (-tf.log(2 * np.pi) + tf.log(self.precs) - self.precs * (diff)))
    return tf.reduce_sum(Eq_logp_x)


class LayerwiseDPMixture(object):

  def __init__(self, weights, uniform=True):
    """
    Args:
    * weights: a list of tensors, weights to be quantized
    """

    self.weights = weights
    self.uniform = uniform
    self.flatten_weights, self.num_weights= flatten(self.weights)
    self.__build_layerwise_dpmm()

  def __build_layerwise_dpmm(self):
    """ Build the variables of dpmm model """
    # self.m0 = []
    self.log_gam_1 = []
    self.log_gam_2 = []
    self.gam_1 = []
    self.gam_2 = []

    self.means = []
    self.precs = []
    self.log_rho = []
    self.r = []
    self.num_sampled = []
    self.loss = 0.

    for layer_id, layer_weight in enumerate(self.flatten_weights):

      layer_num = self.num_weights[layer_id]
      layer_name = self.weights[layer_id].name

      layerwise_dpmm = DPMixture(layer_weight, self.uniform)


  def update_ops(self):
    pass


if __name__ == '__main__':
  # test dpmm class
  os.environ['CUDA_VISIBLE_DEVICES'] = '0'
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  sess = tf.Session(config=config)

  # TODO: cannot find (-3, -1, 1, 3) by dpmm
  # maybe first find clusters (non-uniform), then apply proximal op to make it
  # uniform is a better way?
  weights_1 = np.random.randn(3, 3, 32, 64)*0.1 + 3
  weights_2 = np.random.randn(3, 3, 32, 64)*0.1 - 3
  weights_3 = np.random.randn(3, 3, 32, 64)*0.1 + 1.
  weights_4 = np.random.randn(3, 3, 32, 64)*0.1 - 1.

  weights = np.concatenate((weights_1, weights_2, weights_3, weights_4), axis=2)
  weights = tf.Variable(tf.constant(weights, dtype=tf.float32))

  with tf.variable_scope("DPMM"):
    dpmm = DPMixture(weights, sess, 'uniform')

  sess.run(tf.global_variables_initializer())
  for step in range(1000):
    # update dpmm params
    _, loss = sess.run([dpmm.dpmm_opt, dpmm.loss])

    # proximal update of weights
    top_ind, bits = dpmm.pick_top()
    feed_dict = {dpmm.top_ind: top_ind}
    _, likelihood = sess.run([dpmm.proximal_op, dpmm.top_neg_likelihood], feed_dict=feed_dict)

    if step % 10 == 0:
      tmeans, tpis, precs = sess.run([dpmm.top_means, dpmm.top_pis, dpmm.precs], feed_dict=feed_dict)
      w = sess.run(weights[0,0,:,0])

      print("step: %d, loss: %e" % (step, loss))
      print("likelihood: %e" % likelihood)
      print("bit: %d" % bits)
      print("tpis: ", tpis)
      print("tmeans: ", tmeans, "\t precs: ", precs)
      print(w)
      print("\n")

# Tencent is pleased to support the open source community by making PocketFlow available.
#
# Copyright (C) 2018 THL A29 Limited, a Tencent company. All rights reserved.
#
# Licensed under the BSD 3-Clause License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
""" Util fnctions for Uniform Quantization """

import tensorflow as tf
from tensorflow.contrib import graph_editor as ge
import pdb


def prefix_filter(prefix):
  """ filter out the variable_scope """
  ind = prefix.index('/')
  return prefix[ind+1:]


class UniformQuantization:
  # pylint: disable=too-many-instance-attributes
  """ Class of uniform quantization, now support vanilla, dorefa and pact quantization. """

  def __init__(self, sess, method='vanilla', bucket_size=0, use_buckets=False, bucket_type='split'):
    self.sess = sess
    self.method = method
    self.use_buckets = use_buckets
    self.bucket_size = bucket_size
    self.bucket_type = bucket_type
    self.matmul_ops = []
    self.activation_ops = []
    self.quantized_matmul_ops = []
    self.quantized_activation_ops = []
    self.bucket_storage = tf.constant(0, dtype=tf.int32)  # bits
    self.__safe_check()

    # TODO: add more types of activations and matmuls
    self.support_act_types = ['Relu', 'Relu6', 'Crelu', 'Elu', 'Selu', 'Softplus',\
        'Softsign', 'Sigmoid', 'Tanh']
    self.support_mul_types = ['Conv2D', 'MatMul', 'DepthwiseConv2dNative']

  def insert_quant_op_for_activations(self, act_bit_dict):
    """ Insert quantization operation for activation

    Args:
    * act_bit_dict: A dict with (key: act_op_name, value: act_bits)
    """
    activation_fn = {'Relu': tf.nn.relu,
                     'Tanh': tf.nn.tanh,
                     'Softplus': tf.nn.softplus,
                     'Sigmoid': tf.nn.sigmoid,
                     'Relu6': tf.nn.relu6}
    for op in self.activation_ops:
      old_sgv = ge.sgv(op)
      input_ = old_sgv.inputs[0]

      if op.type in self.support_act_types:
        try:
          tmp_input_ = activation_fn[op.type](input_)
        except KeyError:
          raise NotImplementedError("The activation_fn needs to include %s manually" % op.type)

        prefix = prefix_filter(op.name)
        if self.method == 'vanilla':
          qa = self.__uniform_quantize(tmp_input_, act_bit_dict[op.name], 'activation', prefix)
        elif self.method == 'dorefa':
          qa = self.__dorefa_quantize(tmp_input_,  act_bit_dict[op.name], 'activation', prefix)
        elif self.method == 'pact':
          qa = self.__pact_quantize(tmp_input_, act_bit_dict[op.name], 'activation', prefix)

        tf.add_to_collection("quant_activations", qa)

        # reroute outputs
        new_sgv = ge.sgv(qa.op)
        ge.reroute_outputs(new_sgv, old_sgv)
        self.quantized_activation_ops.append(qa.op)
      else:
        raise ValueError("Unknown activation mode, you may add it manually here")

  def insert_quant_op_for_weights(self, w_bit_dict):
    """Insert quantization operation for weights

    Args:
    * wewight_bit_dict: A dict with (key: matmul_op_name, value: quant_bits)
    """

    for op in self.matmul_ops:
      w = op.inputs[1]
      prefix = prefix_filter(op.name)
      if self.method == 'vanilla':
        qx = self.__uniform_quantize(w, w_bit_dict[op.name], 'weight', prefix)
      elif self.method == 'dorefa':
        qx = self.__dorefa_quantize(w, w_bit_dict[op.name], 'weight', prefix)
      elif self.method == 'pact':
        qx = self.__pact_quantize(w, w_bit_dict[op.name], 'weight', prefix)

      tf.add_to_collection("quant_weights", qx)

      weight_fn = {'MatMul': tf.matmul,
                   'Conv2D': tf.nn.conv2d,
                   'DepthwiseConv2dNative': tf.nn.depthwise_conv2d}
      is_conv_fn = lambda x: 'Conv' in x.type
      try:
        if is_conv_fn(op):
          strides = op.get_attr('strides')
          padding = op.get_attr('padding')
          qx_op = weight_fn[op.type](op.inputs[0], qx, strides, padding).op
        else:
          # fc layers
          qx_op = weight_fn[op.type](op.inputs[0], qx).op
        self.quantized_matmul_ops.append(qx_op)
      except KeyError:
        raise NotImplementedError("Unrecognied Mul op, \
            try to add it into matmul_typs for quantization")

    # replace input
    for wop, qxop in zip(self.matmul_ops, self.quantized_matmul_ops):
      old_sgv = ge.sgv(wop)
      new_sgv = ge.sgv(qxop)
      ge.reroute_inputs(new_sgv, old_sgv)

  def search_matmul_op(self, quantize_all_layers):
    """ search matmul or Conv2D operations in graph for quantization"""

    is_student_fn = lambda x: 'distilled' not in x.name
    for op in self.sess.graph.get_operations():
      if op.type in self.support_mul_types and is_student_fn(op):
        self.matmul_ops.append(op)
    if not quantize_all_layers:
      self.matmul_ops = self.matmul_ops[1:-1] # remain full precision for first and last layer

    return self.matmul_ops

  def search_activation_op(self):
    """ search activation operation in graph for quantization """

    is_student_fn = lambda x: 'distilled' not in x.name
    for op in self.sess.graph.get_operations():
      if op.type in self.support_act_types and is_student_fn(op):
        self.activation_ops.append(op)
    return self.activation_ops

  def get_layerwise_tune_op(self, var, lrn_rate=1e-3):
    """ Get the layerwise fine-tuning ops

    Returns:
    * A list of ops for fine-tuning with len(matmul_ops) elements
    """
    layerwise_diff = []
    tune_ops = []
    for (v, q_op) in zip(var, self.quantized_matmul_ops):
      inputs = q_op.inputs[0]
      quant_outputs = q_op.outputs[0]
      # TODO: wrap it into a function, as also used
      # above.
      if 'MatMul' in q_op.type:
        fp_outputs = tf.matmul(inputs, v)
      elif 'Conv2D' in q_op.type:
        strides = q_op.get_attr('strides')
        padding = q_op.get_attr('padding')
        fp_outputs = tf.nn.conv2d(inputs, v, strides, padding)
      else:
        raise ValueError("Unrecognized Mul Op")

      diff = tf.reduce_mean(tf.square(quant_outputs - fp_outputs))
      tune_ops.append(tf.train.AdamOptimizer(lrn_rate).minimize(diff, var_list=v))
      layerwise_diff.append(diff)
    return tune_ops, layerwise_diff

  def __uniform_quantize(self, x, mbits, mode, prefix=''):
    """Uniform quantization function

    Args:
    * x: A Tensor (weights or activation output)
    * mbits: A scalar Tensor, tf.int64, spicifying number of bit for quantization
    * mode: A string, 'weight' or 'activation', where to quantize
    * prefix: A string, the prefix of scope name

    Returns:
    * A Tensor, uniform quantized value
    """
    if mbits == 32:
      return x

    with tf.variable_scope(prefix + '/quantize'):
      if self.use_buckets and mode == 'weight':
        orig_shape = x.get_shape()
        if self.bucket_type == 'split':
          x, bucket_num, padded_num = self.__split_bucket(x)
        elif self.bucket_type == 'channel':
          x, bucket_num, padded_num = self.__channel_bucket(x)

      g = self.sess.graph
      k = tf.cast(2 ** mbits - 1, tf.float32)

      # clip before scale
      if mode == 'weight':
        x = self.__clip_weights(x)
        with tf.control_dependencies([x]):
          x_normalized, alpha, beta = self.__scale(x, mode)
        with g.gradient_override_map({'Round': 'Identity'}):
          qx = tf.round(x_normalized * k) / k
          qx = self.__inv_scale(qx, alpha, beta)

      elif mode == 'activation':
        x_normalized = tf.clip_by_value(x, 0.0, 1.0)
        with g.gradient_override_map({'Round': 'Identity'}):
          qx = tf.round(x_normalized * k) / k

      if self.use_buckets and mode == 'weight':
        # Reshape w back to the original shape
        qx = tf.reshape(qx, [-1])
        if padded_num != 0:
          qx = tf.reshape(qx[:-padded_num], orig_shape)
        else:
          qx = tf.reshape(qx, orig_shape)
        # Update bucket storage if use buckets.
        self.__updt_bucket_storage(bucket_num)

      print("Quantized: " + tf.get_variable_scope().name)
      return qx

  def __dorefa_quantize(self, x, mbits, mode, prefix=''):
    """ DoReFa quantization function

    Args:
    * x: A Tensor (weights or activation output)
    * mbits: A scalar Tensor, tf.int64, spicifying number of bit for quantization
    * mode: A string, 'weight' or 'activation', where to quantize
    * prefix: A string, the prefix of scope name

    Returns:
    * A Tensor, uniform quantized value
    """
    if mbits == 32:
      return x

    with tf.variable_scope(prefix + '/quantize'):
      if self.use_buckets and mode == 'weight':
        orig_shape = x.get_shape()
        if self.bucket_type == 'split':
          x, bucket_num, padded_num = self.__split_bucket(x)
        elif self.bucket_type == 'channel':
          x, bucket_num, padded_num = self.__channel_bucket(x)

      g = self.sess.graph
      k = tf.cast(2 ** mbits - 1, tf.float32)

      if mode == 'weight':
        # Equation (9) in the paper
        x_normalized = tf.tanh(x) / (2 * tf.reduce_max(tf.abs(tf.tanh(x)))) + 0.5
        with g.gradient_override_map({'Round': 'Identity'}):
          # in DoReFa, qx is in [-1, 1]
          qx = 2 * tf.round(x_normalized * k) / k - 1

      elif mode == 'activation':
        # NOTE: pass through ReLU1() gives worse performance on 32-bit act
        x_normalized = tf.clip_by_value(x, 0.0, 1.0)
        with g.gradient_override_map({'Round': 'Identity'}):
          qx = tf.round(x_normalized * k) / k

      if self.use_buckets and mode == 'weight':
        # Reshape w back to the original shape
        qx = tf.reshape(qx, [-1])
        if padded_num != 0:
          qx = tf.reshape(qx[:-padded_num], orig_shape)
        else:
          qx = tf.reshape(qx, orig_shape)

        # Update bucket storage if use buckets.
        self.__updt_bucket_storage(bucket_num)
      print("DoReFa Quantized: " + tf.get_variable_scope().name)
      return qx

  def __pact_quantize(self, x, mbits, mode, prefix=''):
    """ PACT quantization method, appliable for both weights and act.
    TODO: does not support bucketing for now.

    Args:
    * x: A Tensor (weights or activation output)
    * mbits: A scalar Tensor, tf.int64, spicifying number of bit for quantization
    * mode: A string, 'weight' or 'activation', where to quantize
    * prefix: A string, the prefix of scope name

    Returns:
    * qx: A Tensor, uniform quantized value

    """
    with tf.variable_scope(prefix + '/quantize'):

      if mbits == 32:
        return x

      if mode == 'weight':
        # weight normalize, similar to BN
        # TODO: if bucket is true, x_mean, x_std should be vectors.
        x_mean = tf.stop_gradien(tf.reduce_mean(x))
        x_std = tf.stop_gradien(tf.sqrt(tf.reduce_mean(tf.square(x - x_mean))))
        x_normalized = (x - x_mean) / (x_std + 1e-8)
        scale = tf.get_variable(name='pact_scale', initializer=x_std)
        offset = tf.get_variable(name='pact_offset', initializer=x_mean)

        # initilaize clipping var with 1.5 * x_max
        thresh = tf.get_variable(name='pact_thresh', initializer=tf.constant(3.))
        thresh = tf.clip_by_value(thresh, 1e-3, 10)
        x_pact = tf.clip_by_value(x_normalized, -thresh, thresh)

      elif mode == 'activation':
        # no need to normalize activations since there are BN layers
        thresh = tf.get_variable(name='pact_thresh', initializer=tf.constant(10.))
        thresh = tf.clip_by_value(thresh, 1e-3, 10)
        # x_pact = 0.5 * (tf.abs(x_normalized) - tf.abs(x_normalized - thresh) + thresh)
        x_pact = tf.clip_by_value(x, 0., thresh)
      else:
        raise ValueError("Unknown quantization mode for pact")

      g = self.sess.graph
      k = tf.cast(2 ** mbits - 1, tf.float32)
      with g.gradient_override_map({'Round': 'Identity'}):
        qx = thresh * tf.round(x_pact * k / thresh) / k

      if mode == 'weight':
        # NOTE: scale weights back, according to yuhang, this is not necessary
        qx = scale * qx + offset

      print("PACT Quantized: " + tf.get_variable_scope().name)
      return qx

  def __scale(self, w, mode):
    """linear scale function

    Args:
    * w: A Tensor (weights or activation output),
         the shape is [bucket_size, bucekt_num] if use_buckets else the original size.
    * mode: A string, 'weight' or 'activation'

    Returns:
    * A Tensor, the normalized weights
    * A Tensor, alpha, scalar if activation mode else a vector [bucket_num].
    * A Tensor, beta, scalar if activation mode else a vector [bucket_num].
    """
    if mode == 'weight':
      if self.use_buckets:
        axis = 0
      else:
        axis = None
    elif mode == 'activation':
      axis = None
    else:
      raise ValueError("Unknown mode for scalling")

    w_max = tf.stop_gradient(tf.reduce_max(w, axis=axis))
    w_min = tf.stop_gradient(tf.reduce_min(w, axis=axis))
    eps = tf.constant(value=1e-10, dtype=tf.float32)

    alpha = w_max - w_min + eps
    beta = w_min
    w = (w - beta) / alpha
    return w, alpha, beta

  def __inv_scale(self, w, alpha, beta):
    """Inversed linear scale function

    Args:
    * w: A Tensor (weights or activation output)
    * alpha: A float value, scale factor
    * bete: A float value, scale bias

    Returns:
    * A Tensor, inversed scale value1
    """

    return alpha * w + beta

  def __split_bucket(self, w):
    """Create bucket

    Args:
    * w: A Tensor (weights)

    Returns:
    * A Tensor, with shape [bucket_size, multiple]
    * An integer: the number of buckets
    * An integer, the number of padded elements
    """

    flat_w = tf.reshape(w, [-1])
    num_w = flat_w.get_shape()[0].value
    # use the last value to fill
    fill_value = flat_w[-1]

    multiple, rest = divmod(num_w, self.bucket_size)
    if rest != 0:
      values_to_add = tf.ones(self.bucket_size - rest) * fill_value
      # add the fill_value to make the tensor into a multiple of the bucket size.
      flat_w = tf.concat([flat_w, values_to_add], axis=0)
      multiple += 1

    flat_w = tf.reshape(flat_w, [self.bucket_size, -1])
    padded_num = (self.bucket_size - rest) if rest != 0 else 0

    return flat_w, multiple, padded_num

  def __channel_bucket(self, w):
    """ reshape weights according to bucket for 'channel' type.
        Note that for fc layers, buckets are created row-wisely.
    Args:
      w: A Tensor (weights)

    Returns:
      A Tensor shape [bucket_size, bucket_num], bucket_size = h*w*cin for conv or cin for fc
      A integer: the number of buckets
      A integer (0), zero padded elements
    """
    cout = w.get_shape()[-1].value
    folded_w = tf.reshape(w, [-1, cout])
    return folded_w, cout, 0

  def __safe_check(self):
    """ TODO: Check the name of bucket_type, the value of bucket_size """

    if self.bucket_size < 0:
      raise ValueError("Bucket size must be a postive integer")
    if self.bucket_type not in ['split', 'channel']:
      raise ValueError("Unrecognized bucket type, must be 'weight' or 'channel'.")
    assert self.method in ['vanilla', 'dorefa', 'pact'], NotImplementedError("Unknown uniform quantization method")

  def __updt_bucket_storage(self, bucket_num):
    """ Calculate extra storage for the bucket scalling factors

    Args:
    * bucket_num: a Tensor, the number of buckets, and 2*bucket_num scalling factors
    * alpha: a Tensor, the scalling factor
    """
    self.bucket_storage += bucket_num * 32 * 2 # both alpha and beta, so *2

  # New features
  def __clip_weights(self, x):
    """ Clipping weights
    Args:
    * x: a Tensor, weights or activations
    * mode: a indicator, ['weight', 'activation']
    """
    x_mean = tf.reduce_mean(x)
    x_std = tf.sqrt(tf.reduce_mean(tf.square(x - x_mean)))
    x_max = tf.reduce_max(x)
    x_min = tf.reduce_min(x)
    # by hanyu's trick
    trunc = 3
    ub, lb = tf.where(tf.less_equal(x_mean+trunc*x_std, x_max), x_mean+trunc*x_std, x_max), \
        tf.where(tf.greater_equal(x_mean-trunc*x_std, x_min), x_mean-trunc*x_std, x_min)
    x_clip = tf.clip_by_value(x, lb, ub)
    return x_clip

  def __clip_activations(self, x):
    raise NotImplementedError


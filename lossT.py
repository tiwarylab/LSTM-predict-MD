from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import itertools
import json
import os
import sys
import threading
import weakref

import numpy as np

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session as session_module
from tensorflow.python.eager import context
from tensorflow.python.eager import function as eager_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import device as tfdev
from tensorflow.python.framework import dtypes as dtypes_module
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import ctc_ops as ctc
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import gradients as gradients_module
from tensorflow.python.ops import image_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import tensor_array_grad  # pylint: disable=unused-import
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variables as variables_module
from tensorflow.python.training import server_lib
from tensorflow.python.util import nest
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util import tf_inspect
from tensorflow.python.keras.utils import get_custom_objects

from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.util.tf_export import tf_export


def cast(x, dtype):
  """Casts a tensor to a different dtype and returns it.
  You can cast a Keras variable but it still returns a Keras tensor.
  Arguments:
      x: Keras tensor (or variable).
      dtype: String, either (`'float16'`, `'float32'`, or `'float64'`).
  Returns:
      Keras tensor with dtype `dtype`.
  Examples:
      Cast a float32 variable to a float64 tensor
  ```python
      >>> import tensorflow as tf
      >>> from tensorflow.keras import backend as K
      >>> input = K.ones(shape=(1,3))
      >>> print(input)
      >>> cast_input = K.cast(input, dtype='float64')
      >>> print(cast_input)
      <tf.Variable 'Variable:0' shape=(1, 3) dtype=float32,
           numpy=array([[1., 1., 1.]], dtype=float32)>
      tf.Tensor([[1. 1. 1.]], shape=(1, 3), dtype=float64)
  ```
  """
  return math_ops.cast(x, dtype)


def flatten(x):
  """Flatten a tensor.
  Arguments:
      x: A tensor or variable.
  Returns:
      A tensor, reshaped into 1-D
  """
  return array_ops.reshape(x, [-1])


def sparse_categorical_crossentropy(target, output, from_logits=False, axis=-1):
  """Categorical crossentropy with integer targets.
  Arguments:
      target: An integer tensor.
      output: A tensor resulting from a softmax
          (unless `from_logits` is True, in which
          case `output` is expected to be the logits).
      from_logits: Boolean, whether `output` is the
          result of a softmax, or is a tensor of logits.
      axis: Int specifying the channels axis. `axis=-1` corresponds to data
          format `channels_last', and `axis=1` corresponds to data format
          `channels_first`.
  Returns:
      Output tensor.
  Raises:
      ValueError: if `axis` is neither -1 nor one of the axes of `output`.
  """
  if not from_logits:
    if (isinstance(output, (ops.EagerTensor, variables_module.Variable)) or
        output.op.type != 'Softmax'):
      epsilon_ = _constant_to_tensor(epsilon(), output.dtype.base_dtype)
      output = clip_ops.clip_by_value(output, epsilon_, 1 - epsilon_)
      output = math_ops.log(output)
    else:
      # When softmax activation function is used for output operation, we
      # use logits from the softmax function directly to compute loss in order
      # to prevent collapsing zero when training.
      # See b/117284466
      assert len(output.op.inputs) == 1
      output = output.op.inputs[0]

  rank = len(output.shape)
  axis = axis % rank
  if axis != rank - 1:
    permutation = list(range(axis)) + list(range(axis + 1, rank)) + [axis]
    output = array_ops.transpose(output, perm=permutation)

  output_shape = output.shape
  targets = cast(flatten(target), 'int64')
  logits = array_ops.reshape(output, [-1, int(output_shape[-1])])
  res = nn.sparse_softmax_cross_entropy_with_logits(
      labels=targets, logits=logits)
  if len(output_shape) >= 3:
    # If our output includes timesteps or spatial dimensions we need to reshape
    return array_ops.reshape(res, array_ops.shape(output)[:-1])
  else:
    return res



# Doing this allows you to then to refer to your custom object via string.
get_custom_objects().update({'cast':cast,
                             'sparse_categorical_crossentropy': sparse_categorical_crossentropy})

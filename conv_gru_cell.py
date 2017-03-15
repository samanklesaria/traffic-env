import tensorflow as tf
import tensorflow.contrib.layers as tl
from  tensorflow.contrib.rnn import RNNCell

class ConvGRUCell(RNNCell):
  def __init__(self, hidden_channels, dims):
    self._output_size = tf.TensorShape([*dims, hidden_channels])
    self.dims = dims
    self.hidden_channels = hidden_channels
  
  @property
  def output_size(self): return self._output_size

  @property
  def state_size(self): return self._output_size

  def __call__(self, inputs, state, scope=None):
    with tf.variable_scope(scope or type(self).__name__):
      input_and_hidden = tf.concat([state, inputs], 3)
      z = tl.conv2d(input_and_hidden, self.hidden_channels, self.dims,
          biases_initializer=None, activation_fn=tf.sigmoid)
      r =  tl.conv2d(input_and_hidden, self.hidden_channels, self.dims,
          biases_initializer=None, activation_fn=tf.sigmoid)
      input_and_updated = tf.concat([r * state, inputs], 3)
      h_tilde = tl.conv2d(input_and_updated, self.hidden_channels, self.dims,
          biases_initializer=None, activation_fn=tf.tanh)
      h = (1 - z) * state + z * h_tilde
      return h, h




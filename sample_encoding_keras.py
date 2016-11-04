import tensorflow as tf
from keras.layers import Dense, Embedding, Lambda
from keras.objectives import binary_crossentropy
from keras.metrics import binary_accuracy as accuracy
from keras import backend as K
from keras import activations, initializations, regularizers, constraints
from keras.engine.topology import Layer, InputSpec
from keras.layers.wrappers import Wrapper
import numpy as np
K.set_learning_phase(1)


def get_pair_indices(sequence_length):
    """get indices for each pair in a sample x"""
    pair_indices = []
    for i in range(sequence_length):
        for j in range(i + 1, sequence_length):
            pair_indices.append([i, j])
    return pair_indices


def get_batch_pair_indices(batch_size, sequence_length):
    """get the indices for a batch"""
    indices = get_pair_indices(sequence_length)
    comb_num = len(indices)
    batch_indices = []
    for i in range(batch_size):
        for j in range(len(indices)):
            batch_indices.append([[i, indices[j][0]], [i, indices[j][1]]])
    batch_indices = np.array(batch_indices)
    return (batch_indices[:, 0].reshape(batch_size, comb_num, 2),
            batch_indices[:, 1].reshape(batch_size, comb_num, 2))


def _gate(c1, c2, gate_type='sum'):
    """pair interaction method"""
    print("Using gate: ", gate_type)
    if gate_type == 'sum':
        return tf.add(c1, c2)
    if gate_type == 'mul':
        return tf.mul(c1, c2)
    if gate_type == 'avg':
        return  tf.add(c1, c2) * 0.5
    #if gate_type == 'highway':
    #    return flat_highway_gate_layer(c1, c2)
    #if gate_type == 'sum_tanh':
    #    return tf.tanh(tf.add(c1, c2))
    #if gate_type == 'mul_tanh':
    #    return tf.tanh(tf.mul(c1, c2))
    #if gate_type == 'p_norm':
    #    p = 2
    #    return tf.pow(tf.add(tf.pow(c1, p), tf.pow(c2, p)), 1 / p)


def _norm(interactions, norm_type='l2'):
    """metrics for max-pooling"""
    if norm_type == 'l2':
        return tf.reduce_sum(tf.square(interactions), 2)
    if norm_type == 'l1':
        return tf.reduce_sum(tf.abs(interactions), 2)
    if norm_type == 'avg':
        return tf.reduce_sum(interactions, 2)

def gather_nd(params, indices):
    """
        just work for 3D tensor with 2D indices
        ref: cruvadom's implementation for gather_nd with some modifications,
             https://github.com/tensorflow/tensorflow/issues/206
        e.g:
            params = [[[1,3],[2,2]]]
            indices = [[0,1]]
            gather_nd(params, indices) returns [[[2,2]]]
    """
    shape = params.get_shape().as_list()
    if type(indices) is tf.Tensor:
        indices_shape = indices.get_shape().as_list()
    elif type(indices) is np.ndarray:
        indices_shape = indices.shape
    else:
        raise ValueError("indices should be np.ndarray or tf.Tensor")

    rank = len(shape)
    flat_params = tf.reshape(params, [-1])
    multipliers = [reduce(lambda x, y: x * y, shape[i + 1:], 1)
                   for i in range(0, rank)]
    # print(multipliers)
    indices_unpacked = tf.unpack(tf.transpose(
        indices, [rank - 1] + range(0, rank - 1)))
    flat_indices = sum([a * b for a, b in zip(multipliers, indices_unpacked)])
    flat_indices = tf.expand_dims(flat_indices, 2)
    flat_indices = tf.tile(flat_indices, [1, 1, shape[-1]])
    flat_indices_shape = flat_indices.get_shape().as_list()
    indices_3d = tf.constant(flat_indices_shape[0] * flat_indices_shape[1] * range(shape[-1]), shape=flat_indices_shape)
    flat_indices = tf.add(tf.to_int32(flat_indices), indices_3d)
    flat_indices = tf.reshape(flat_indices, [-1])
    gathered = tf.gather(flat_params, flat_indices)
    # return flat_indices
    return tf.reshape(gathered, [shape[0], indices_shape[1], shape[-1]])


def _get_max_pooling_indices(batch_size, indices, k):
    """
        refers from tianyao's code
        get the indices for the max-pooling
    """
    batch_indices = tf.range(batch_size)
    batch_indices = tf.expand_dims(batch_indices, 1)
    batch_indices = tf.expand_dims(batch_indices, 2)
    batch_indices = tf.tile(batch_indices, [1, k, 1])
    indices = tf.expand_dims(indices, 2)
    return tf.concat(2, [batch_indices, indices])


def _pairwise_indices(sequence_length):
    left = [i for i in range(sequence_length) for j in range(i + 1, sequence_length)]
    right = [j for i in range(sequence_length) for j in range(i + 1, sequence_length)]
    return left, right


def _gather_interaction(x, indices):
    shape = x.get_shape().as_list()
    
    batch_size = shape[0]
    sequence_length = shape[1]
    interaction_number = len(indices)
    
    indices = np.array([i * sequence_length + j for i in range(batch_size) for j in indices])
    
    x = tf.reshape(x, [-1, ] + shape[2:])
    x = tf.gather(x, indices)
    return tf.reshape(x, [-1, interaction_number, ] + shape[2:])


class PairwiseInteraction(Layer):
    def __init__(self, gate_type='sum', activation='linear', dropout=0., **kwargs):
        self.gate_type = gate_type
        self.input_spec = [InputSpec(ndim=3)]
        self.activation = activations.get(activation)
        self.dropout = dropout
        super(PairwiseInteraction, self).__init__(**kwargs)

    def build(self, input_shape):
        self.batch_size = input_shape[0]
        self.sequence_length = input_shape[1]
        self.embedding_size = input_shape[2]
        self.interaction_number  = int(0.5 * (self.sequence_length ** 2 - self.sequence_length))
        self.first_indices, self.second_indices = \
            get_batch_pair_indices(self.batch_size, self.sequence_length)

    def call(self, x, mask=None):
        first_element = gather_nd(x, self.first_indices)
        second_element = gather_nd(x, self.second_indices)
        interactions = _gate(first_element, second_element, self.gate_type)
        if 0. < self.dropout < 1.:
            retain_p = 1. - self.dropout
            B = K.random_binomial((self.batch_size, self.interaction_number, self.embedding_size), p=retain_p) * (1. / retain_p)
            # B = K.expand_dims(B)
            interactions = K.in_train_phase(interactions * B, interactions)
        return self.activation(interactions)

    def get_output_shape_for(self, input_shape):
        return (self.batch_size, self.interaction_number, self.embedding_size)

    def get_config(self):
        config = {'gate_type': self.gate_type,
                  'activation': self.activation.__name__}
        base_config = super(PairwiseInteraction, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    

class PairGateInteraction(PairwiseInteraction):
    def build(self, input_shape):
        super(PairGateInteraction, self).build(input_shape[0])

    def call(self, input_list, mask=None):
        first_element = gather_nd(input_list[0], self.first_indices)
        second_element = gather_nd(input_list[1], self.second_indices)
        interactions = _gate(first_element, second_element, self.gate_type)
        if 0. < self.dropout < 1.:
            retain_p = 1. - self.dropout
            B = K.random_binomial((self.batch_size, self.interaction_number, self.embedding_size), p=retain_p) * (1. / retain_p)
            # B = K.expand_dims(B)
            interactions = K.in_train_phase(interactions * B, interactions)
        return self.activation(interactions)
    
    def get_output_shape_for(self, input_shape):
        return super(PairGateInteraction, self).get_output_shape_for(input_shape[0])


class KMaxTensorPooling(Layer):
    def __init__(self, k, norm_type='l2', **kwargs):
        self.k = k
        self.norm_type = norm_type
        super(KMaxTensorPooling, self).__init__(**kwargs)

    def build(self, input_shape):
        self.batch_size = input_shape[0]
        self.interaction_number = input_shape[1]
        self.embedding_size = input_shape[2]

    def call(self, x, mask=None):
        norms = _norm(x, self.norm_type)
        _, max_indices = tf.nn.top_k(norms, self.k)
        self.pooling_indices = _get_max_pooling_indices(self.batch_size, max_indices, self.k)
        pooling_rst = gather_nd(x, self.pooling_indices)
        return pooling_rst

    def get_output_shape_for(self, input_shape):
        return (self.batch_size, self.k, self.embedding_size)

    def get_config(self):
        config = {'k': self.k,
                  'norm_type': self.norm_type}
        base_config = super(KMaxTensorPooling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class FM(Layer):
    input_ndim = 2

    def __init__(self, input_dim, hidden_dim,
                 init='uniform', input_length=None,
                 activity_regularizer=None,
                 w_regularizer=None, w_constraint=None,
                 V_regularizer=None, V_constraint=None,
                 b_regularizer=None, b_constraint=None,
                 bias=True, weights=None,
                 dropout=0., **kwargs):

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.init = initializations.get(init)
        self.input_length = input_length
        self.dropout = dropout
        self.bias = bias

        self.w_constraint = constraints.get(w_constraint)
        self.w_regularizer = regularizers.get(w_regularizer)
        self.V_constraint = constraints.get(V_constraint)
        self.V_regularizer = regularizers.get(V_regularizer)
        self.b_constraint = constraints.get(b_constraint)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.initial_weights = weights
        if 0. < self.dropout < 1.:
            self.uses_learning_phase = True
        kwargs['input_shape'] = (self.input_length,)
        kwargs['input_dtype'] = 'int32'
        super(FM, self).__init__(**kwargs)

    def build(self, input_shape):
        self.w = self.init((self.input_dim, 1),
                           name='{}_w'.format(self.name))
        self.V = self.init((self.input_dim, self.hidden_dim),
                           name='{}_V'.format(self.name))
        if self.bias:
            self.b = K.zeros((1,),
                             name='{}_b'.format(self.name))
            self.trainable_weights = [self.w, self.V, self.b]
        else:
            self.trainable_weights = [self.w, self.V]

        self.constraints = {}
        if self.w_constraint:
            self.constraints[self.w] = self.w_constraint
        if self.V_constraint:
            self.constraints[self.V] = self.V_constraint
        if self.b_constraint:
            self.constraints[self.b] = self.b_constraint

        self.regularizers = []
        if self.w_regularizer:
            self.w_regularizer.set_param(self.w)
            self.regularizers.append(self.w_regularizer)
        if self.V_regularizer:
            self.V_regularizer.set_param(self.V)
            self.regularizers.append(self.V_regularizer)
        if self.b_regularizer:
            self.b_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)

        if self.activity_regularizer:
            self.activity_regularizer.set_layer(self)
            self.regularizers.append(self.activity_regularizer)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0],1)

    def call(self, x, mask=None):
        if K.dtype(x) != 'int32':
            x = K.cast(x, 'int32')
        if 0. < self.dropout < 1.:
            retain_p = 1. - self.dropout
            B = K.random_binomial((self.input_dim,), p=retain_p) * (1. / retain_p)
            B = K.expand_dims(B)
            V = K.in_train_phase(self.V * B, self.V)
        else:
            V = self.V
        _w = K.gather(self.w, x)
        output = tf.reduce_sum(_w,1)
        if self.bias:
            output += self.b
        v = K.gather(self.V, x)
        _v = (tf.reduce_sum(tf.square(tf.reduce_sum(v,2)) - tf.reduce_sum(tf.square(v),2),1) * 0.5)
        output += tf.reshape(_v, [-1,1])
        return tf.sigmoid(output)

    def get_config(self):
        config = {'input_dim': self.input_dim,
                  'hidden_dim': self.hidden_dim,
                  'init': self.init.__name__,
                  'input_length': self.input_length,
                  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  'w_regularizer': self.w_regularizer.get_config() if self.w_regularizer else None,
                  'w_constraint': self.w_constraint.get_config() if self.w_constraint else None,
                  'V_regularizer': self.V_regularizer.get_config() if self.V_regularizer else None,
                  'V_constraint': self.V_constraint.get_config() if self.V_constraint else None,
                  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                  'b_constraint': self.b_constraint.get_config() if self.b_constraint else None,
                  'dropout': self.dropout}
        base_config = super(FM, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    
    
class LR(Layer):

    def __init__(self, input_dim,
                 init='uniform', input_length=None,
                 activity_regularizer=None,
                 w_regularizer=None, w_constraint=None,
                 b_regularizer=None, b_constraint=None,
                 bias=True, weights=None,
                 dropout=0., **kwargs):

        self.input_dim = input_dim
        self.init = initializations.get(init)
        self.input_length = input_length
        self.dropout = dropout
        self.bias = bias

        self.w_constraint = constraints.get(w_constraint)
        self.w_regularizer = regularizers.get(w_regularizer)
        self.b_constraint = constraints.get(b_constraint)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.initial_weights = weights
        if 0. < self.dropout < 1.:
            self.uses_learning_phase = True
        kwargs['input_shape'] = (self.input_length,)
        kwargs['input_dtype'] = 'int32'
        super(LR, self).__init__(**kwargs)

    def build(self, input_shape):
        self.w = self.init((self.input_dim, 1),
                           name='{}_w'.format(self.name))
        if self.bias:
            self.b = K.zeros((1,),
                             name='{}_b'.format(self.name))
            self.trainable_weights = [self.w, self.b]
        else:
            self.trainable_weights = [self.w]

        self.constraints = {}
        if self.w_constraint:
            self.constraints[self.w] = self.w_constraint
        if self.b_constraint:
            self.constraints[self.b] = self.b_constraint

        self.regularizers = []
        if self.w_regularizer:
            self.w_regularizer.set_param(self.w)
            self.regularizers.append(self.w_regularizer)
        if self.b_regularizer:
            self.b_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)

        if self.activity_regularizer:
            self.activity_regularizer.set_layer(self)
            self.regularizers.append(self.activity_regularizer)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0],1)

    def call(self, x, mask=None):
        if K.dtype(x) != 'int32':
            x = K.cast(x, 'int32')
        if 0. < self.dropout < 1.:
            retain_p = 1. - self.dropout
            B = K.random_binomial((self.input_dim,), p=retain_p) * (1. / retain_p)
            B = K.expand_dims(B)
            w = K.in_train_phase(self.w * B, self.w)
        else:
            w = self.w
        _w = K.gather(w, x)
        output = tf.reduce_sum(_w,1)
        if self.bias:
            output += self.b
        output += tf.reshape(output, [-1,1])
        return tf.sigmoid(output)

    def get_config(self):
        config = {'input_dim': self.input_dim,
                  'init': self.init.__name__,
                  'input_length': self.input_length,
                  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  'w_regularizer': self.w_regularizer.get_config() if self.w_regularizer else None,
                  'w_constraint': self.w_constraint.get_config() if self.w_constraint else None,
                  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                  'b_constraint': self.b_constraint.get_config() if self.b_constraint else None,
                  'dropout': self.dropout}
        base_config = super(LR, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    
class TimeDistributed(Wrapper):
    """This wrapper allows to apply a layer to every
    temporal slice of an input.
    The input should be at least 3D,
    and the dimension of index one will be considered to be
    the temporal dimension.
    Consider a batch of 32 samples, where each sample is a sequence of 10
    vectors of 16 dimensions. The batch input shape of the layer is then `(32, 10, 16)`
    (and the `input_shape`, not including the samples dimension, is `(10, 16)`).
    You can then use `TimeDistributed` to apply a `Dense` layer to each of the 10 timesteps, independently:
    ```python
        # as the first layer in a model
        model = Sequential()
        model.add(TimeDistributed(Dense(8), input_shape=(10, 16)))
        # now model.output_shape == (None, 10, 8)
        # subsequent layers: no need for input_shape
        model.add(TimeDistributed(Dense(32)))
        # now model.output_shape == (None, 10, 32)
    ```
    The output will then have shape `(32, 10, 8)`.
    Note this is strictly equivalent to using `layers.core.TimeDistributedDense`.
    However what is different about `TimeDistributed`
    is that it can be used with arbitrary layers, not just `Dense`,
    for instance with a `Convolution2D` layer:
    ```python
        model = Sequential()
        model.add(TimeDistributed(Convolution2D(64, 3, 3), input_shape=(10, 3, 299, 299)))
    ```
    # Arguments
        layer: a layer instance.
    """
    def __init__(self, layer, **kwargs):
        self.supports_masking = True
        super(TimeDistributed, self).__init__(layer, **kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 3
        self.input_spec = [InputSpec(shape=input_shape)]
        child_input_shape = (input_shape[0],) + input_shape[2:]
        if not self.layer.built:
            self.layer.build(child_input_shape)
            self.layer.built = True
        super(TimeDistributed, self).build()

    def get_output_shape_for(self, input_shape):
        child_input_shape = (input_shape[0],) + input_shape[2:]
        child_output_shape = self.layer.get_output_shape_for(child_input_shape)
        timesteps = input_shape[1]
        return (child_output_shape[0], timesteps) + child_output_shape[1:]

    def call(self, X, mask=None):
        input_shape = self.input_spec[0].shape
        # no batch size specified, therefore the layer will be able
        # to process batches of any size
        # we can go with reshape-based implementation for performance
        input_length = input_shape[1]
        if not input_length:
            input_length = K.shape(X)[1]
        X = K.reshape(X, (-1, ) + input_shape[2:])  # (nb_samples * timesteps, ...)
        y = self.layer.call(X)  # (nb_samples * timesteps, ...)
        # (nb_samples, timesteps, ...)
        output_shape = self.get_output_shape_for(input_shape)
        y = K.reshape(y, (-1, input_length) + output_shape[2:])
        return y
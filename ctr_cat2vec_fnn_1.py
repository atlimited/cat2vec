import numpy as np
import pandas as pd
from time import time
from sklearn import metrics
from sample_encoding_keras import *
from utility import Options, ctr_batch_generator, read_feat_index, eval_auc
import tensorflow as tf

opts = Options()
print('Loading data...')

#############################
# Settings for ipinyou
#############################
opts.sequence_length = 22
opts.total_training_sample = 127127
# change here for different camp
# 1458 2261 2997 3386 3476 2259 2821 3358 3427 all
BASE_PATH = './data/make-ipinyou-data/all/'
opts.field_indices_path = BASE_PATH + 'field_indices.txt'
opts.train_path = BASE_PATH + 'train.yzx.10.txt'
opts.test_path = BASE_PATH + 'test.yzx.txt'
opts.featindex = BASE_PATH + 'featindex.txt'
opts.test_num = 0
opts.model_name = 'ipinyou_distcrimination_concat'

### Paramerters for tuning
opts.gate_type = 'sum' # sum, mul
opts.activation_type = 'tanh' # sigmoid, tanh, relu, linear etc.
opts.dropout = 0.05
opts.batch_size = 64
opts.embedding_size = 16
opts.interaction_times = 1
opts.epochs_to_train = 5
opts.highway = False




gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.45)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
K.set_session(sess)
id2cate, cate2id, opts.vocabulary_size = read_feat_index(opts)

#############################
# Model
#############################
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Embedding, Dropout, Activation, Flatten, Merge, merge, Highway
# from keras.layers.wrappers import TimeDistributed
# Note, there is a bug in original implementation of TimeDistributed in keras
input_tensor = tf.placeholder(tf.int32, shape=(
    opts.batch_size, opts.sequence_length))
a = Input(batch_shape=(opts.batch_size, opts.sequence_length),
          tensor=input_tensor, name='input')
e = Embedding(opts.vocabulary_size,
              opts.embedding_size,
              input_length=opts.sequence_length, name='embedding')(a)
x = e
for i in range(1, opts.interaction_times + 1):
    x = PairwiseInteraction(gate_type=opts.gate_type,
                            activation=opts.activation_type,
                            dropout=opts.dropout,
                            name='interaction layer %d' % i)(x)
    if opts.highway:
        x = TimeDistributed(Highway())(x)
    x = KMaxTensorPooling(opts.sequence_length,
                          name='kmaxpooling layer %d' % i)(x)
x = Lambda(lambda t: tf.reshape(
    t, [-1, opts.sequence_length * opts.embedding_size]))(x)
e = Lambda(lambda t: tf.reshape(
    t, [-1, opts.sequence_length * opts.embedding_size]))(e)
x = merge([x, e], mode='concat', concat_axis=1)
x = Dense(128, activation='sigmoid')(x)
x = Dense(32, activation='sigmoid')(x)
x = Dense(1, activation='sigmoid', name='prob')(x)

model = Model(input=a, output=x)
model.compile(optimizer='nadam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
print(model.summary())

opts.current_epoch = 1
target = [model.get_layer(name='prob').output]
get_prob = K.function([input_tensor], target)

#############################
# Model Training
#############################
for i in range(opts.epochs_to_train):
    print("Current Global Epoch: ", opts.current_epoch)
    training_batch_generator = ctr_batch_generator(opts)
    history = model.fit_generator(
        training_batch_generator, opts.total_training_sample, 1)
    if i % 1 == 0:
        model.save('model_' + opts.model_name + '_' +
                   str(opts.current_epoch) + '.h5')
        eval_auc(model, opts, target=target, get_prob=get_prob)
    opts.current_epoch += 1

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
opts.interaction_times = 1
opts.total_training_sample = 127127
# change here for different camp
# 1458 2261 2997 3386 3476 2259 2821 3358 3427 all
BASE_PATH = './data/make-ipinyou-data/all/'
opts.field_indices_path = BASE_PATH + 'field_indices.txt'
opts.train_path = BASE_PATH + 'train.yzx.10.txt'
opts.test_path = BASE_PATH + 'test.yzx.txt'
opts.featindex = BASE_PATH + 'featindex.txt'
opts.model_name = 'ipinyou_fm'

### Paramerters for tuning
opts.batch_size = 256
opts.embedding_size = 8
#opts.learning_rate = 0.1


gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
K.set_session(sess)
id2cate, cate2id, opts.vocabulary_size = read_feat_index(opts)


#############################
# Model
#############################
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Embedding, Dropout, Activation, Flatten, Merge, merge
from keras.layers import Convolution1D, MaxPooling1D

model = Sequential()
model.add(FM(opts.vocabulary_size,opts.embedding_size,input_length=opts.sequence_length,dropout=0.1, name='prob'))
model.compile(optimizer='nadam',
      loss='binary_crossentropy',
      metrics=['accuracy'])
print(model.summary())
opts.current_epoch = 1


#############################
# Model Training
#############################
for i in range(opts.epochs_to_train):
    print("Current Global Epoch: ", opts.current_epoch)
    training_batch_generator = ctr_batch_generator(opts)
    history = model.fit_generator(training_batch_generator, opts.total_training_sample, 1)
    if i % 1 == 0:
        opts.batch_size = 4096
        model.save('model_'+opts.model_name+'_' + str(opts.current_epoch) + '.h5')
        eval_auc(model, opts)
    opts.current_epoch += 1

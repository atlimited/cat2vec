from __future__ import print_function
import numpy as np
from sample_encoding_keras import *
from collections import Counter

import tensorflow.contrib.learn as skflow
from numpy.linalg import inv
import emcee


def lnprob_trunc_norm(x, mean, bounds, C):
    if np.any(x < bounds[:, 0]) or np.any(x >= bounds[:, 1]):
        return -np.inf
    else:
        return -0.5 * (x - mean).dot(inv(C)).dot(x - mean)


def get_MTND(opts):
    Nwalkers = 10
    Ndim = 4
    Nsteps = 1
    mean = opts.mean
    cov = opts.cov
    # cov = np.array([[5,1,1,1],[1,5,1,2],[2,1,5,1],[1,3,1,5]])
    bounds = np.array([[0, 10], [0, 10], [0, 10], [0, 10]])
    pos = emcee.utils.sample_ball(mean, np.sqrt(np.diag(cov)), size=Nwalkers)
    S = emcee.EnsembleSampler(
        Nwalkers, Ndim, lnprob_trunc_norm, args=(mean, bounds, cov))
    while True:
        pos, prob, state = S.run_mcmc(pos, Nsteps)
        for x in pos:
            yield x


class Options(object):
    """Options used by  the model."""

    def __init__(self):
        # Model options
        self.embedding_size = 32
        self.learning_rate = 1.
        self.epochs_to_train = 100
        self.batch_size = 128


def batch_generator(opts):
    MTND_generator = get_MTND(opts)
    while True:
        batch = []
        labels = []
        for _ in xrange(opts.batch_size):
            sample = []
            target = 0.
            if np.random.rand() > 0.5:
                X = np.clip(MTND_generator.next(), 0, 9)
                X = map(int, X)
                target = 1.
            else:
                X = map(int, np.random.rand(opts.sequence_length) * 10)
            batch.append([i * 10 + X[i] for i in xrange(opts.sequence_length)])
            labels.append(target)
        yield batch, labels


from keras.models import Sequential, Model
from keras.layers import Input, Dense, Embedding, Dropout, Activation, Flatten


def get_model(opts):
    input_tensor = tf.placeholder(tf.int32, shape=(
        opts.batch_size, opts.sequence_length))
    a = Input(batch_shape=(opts.batch_size, opts.sequence_length),
              tensor=input_tensor, name='input')
    x = Embedding(opts.feat_size,
                  opts.embedding_size,
                  input_length=opts.sequence_length, name='embedding')(a)
    for i in range(1, opts.interaction_times + 1):
        x = PairwiseInteraction(gate_type='mul', activation='linear',
                                dropout=opts.dropout, name='interaction layer %d' % i)(x)
        x = KMaxTensorPooling(opts.sequence_length,
                              name='kmaxpooling layer %d' % i)(x)
    x = Lambda(lambda t: tf.reshape(
        t, [-1, opts.sequence_length * opts.embedding_size]))(x)
    x = Dense(1, activation='sigmoid', name='prob')(x)
    model = Model(input=a, output=x)
    model.compile(optimizer='nadam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    # print(model.summary())
    return model, input_tensor


# In[17]:
def run():
    opts = Options()
    opts.batch_size = 16
    opts.learning_rate = 0.01
    opts.l2_reg_lambda = 0.
    opts.sequence_length = 4
    opts.embedding_size = 8
    opts.interaction_times = 2
    opts.feat_size = opts.sequence_length * 10
    opts.lower, opts.upper = 0, 10
    opts.mu, opts.sigma = 5, 5
    opts.mean = np.random.rand(4) * 10
    opts.cov = np.random.rand(4, 4) * 5
    opts.dropout = 0.20

    model, input_tensor = get_model(opts)
    opts.current_epoch = 1
    # training_batch_generator.next()

    for i in range(5):
        print("Current Global Epoch: ", opts.current_epoch)
        training_batch_generator = batch_generator(opts)
        history = model.fit_generator(
            training_batch_generator, 10000, 1, verbose=0)
        opts.current_epoch += 1

    def get_maxpooling_indices(model, input_tensor, X_batch, layers=['kmaxpooling layer 1']):
        targets = [model.get_layer(
            name=layer).pooling_indices for layer in layers]
        get_pooling_indices = K.function([input_tensor], targets)
        pooling_indices = get_pooling_indices([X_batch])
        return pooling_indices
    training_batch_generator = batch_generator(opts)
    layers = ['kmaxpooling layer %d' %
              i for i in range(1, opts.interaction_times + 1)]
    pair_indices = get_pair_indices(opts.sequence_length)

    from itertools import combinations
    all_pairs = []
    all_pairs_3 = []
    all_combination = []
    all_combination_3 = []

    for _ in range(1000):
        X_batch, _ = training_batch_generator.next()
        X_batch_1 = []
        for x in X_batch:
            all_combination.extend(list(combinations(x, 2)))
            all_combination_3.extend(list(combinations(x, 3)))
        pooling = get_maxpooling_indices(
            model, input_tensor, X_batch, layers=layers)
        batch_pooling = pooling[0]
        batch_pooling_3 = pooling[1]
        pool_rst = ''
        for i in range(len(batch_pooling)):
            sample = batch_pooling[i]
            x_temp = []
            # print(sample)
            for pair in sample:
                # print(pair)
                str_pair = str(X_batch[i][pair_indices[pair[1]][
                               0]]) + ',' + str(X_batch[i][pair_indices[pair[1]][1]])
                all_pairs.append(str_pair)
                pool_rst += str_pair + '\t'
                x_temp.append(str_pair)
            X_batch_1.append(x_temp)
            pool_rst += '\n'
        for i in range(len(batch_pooling_3)):
            sample = batch_pooling_3[i]
            for pair in sample:
                str_pair = str(X_batch_1[i][pair_indices[pair[1]][
                               0]]) + ',' + str(X_batch_1[i][pair_indices[pair[1]][1]])
                str_pair = ','.join(sorted(list(set(str_pair.split(',')))))
                all_pairs_3.append(str_pair)

    # In[23]:

    correct_3 = Counter([str(a[0]) + ',' + str(a[1]) + ',' + str(a[2])
                         for a in all_combination_3]).most_common(20)
    correct_3 = [(c[0], c[1], i) for i, c in enumerate(correct_3)]
    correct_3

    # In[34]:

    rsts_3 = Counter(all_pairs_3).most_common(100)
    rsts_pair_3 = [c[0] for c in rsts_3][0:20]
    rsts_3 = {c[0]: i for i, c in enumerate(rsts_3)}
    rsts_3 = [rsts_3.get(c[0]) for c in correct_3]
    def convert(x):
        if r is not None:
            return r
        else:
            return 100
    rsts_3 = [convert(r) for r in rsts_3]
    rsts_3

    # In[35]:

    correct = Counter([str(a[0]) + ',' + str(a[1])
                       for a in all_combination]).most_common(20)
    correct = [(c[0], i) for i, c in enumerate(correct)]
    correct

    # In[36]:

    rsts = Counter(all_pairs).most_common(100)
    rsts_pair = [c[0] for c in rsts][0:20]
    rsts = {c[0]: i for i, c in enumerate(rsts)}
    rsts = [rsts.get(c[0]) for c in correct]
    def convert(x):
        if r is not None:
            return r
        else:
            return 100
    rsts = [convert(r) for r in rsts]
    import scipy
    r_2 = scipy.stats.spearmanr(range(1, 21), rsts).correlation
    r_3 = scipy.stats.spearmanr(range(1, 21), rsts_3).correlation
    #print('rank correlation 2: ', r_2)
    #print('rank correlation 3: ', r_3)

    intersections = set([c[0] for c in correct]).intersection(set(rsts_pair))
    recall = 1. * len(intersections) / len(correct)
    precision_2 = 1. * len(intersections) / len(rsts)
    #print('recall:', recall)
    #print('precision 2:', precision_2)
    intersections = set([c[0] for c in correct_3]
                        ).intersection(set(rsts_pair_3))
    recall = 1. * len(intersections) / len(correct_3)
    precision_3 = 1. * len(intersections) / len(rsts_3)
    #print('recall for triple:', recall)
    #print('precision for triple 3:', precision_3)
    return np.array([r_2, r_3, precision_2, precision_3])

results = np.zeros(4)
n = 10
for i in range(n):
    results += run()
results = results / n


print({k: v for k, v in zip(
    ['r_2', 'r_3', 'precision_2', 'precision_3'], results)})

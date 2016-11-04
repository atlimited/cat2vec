from __future__ import print_function
import numpy as np
import pandas as pd
from sklearn import metrics

class Options(object):
    """Options used by  the model."""

    def __init__(self):
        # Model options.

        # Embedding dimension.
        self.embedding_size = 32

        # The initial learning rate.
        self.learning_rate = 1.

        # Number of epochs to train. After these many epochs, the learning
        # rate decays linearly to zero and the training stops.
        self.epochs_to_train = 100

        # Number of examples for one training step.
        self.batch_size = 128
        
        self.log_path = './ctr.log'
        
def read_file(path, infinite=True):
    while True:
        fi = open(path,'r')
        for line in fi:
            yield map(int,line.replace('\n', '').split(' '))
        if infinite == False:
            break
    yield None

def ctr_batch_generator(opts, train=True):
    if train:
        file_reader = read_file(opts.train_path, True)
    else:
        file_reader = read_file(opts.test_path, False)
    while True:
        batch = np.ndarray(shape=(opts.batch_size, opts.sequence_length))
        labels = np.ndarray(shape=(opts.batch_size))
        for i in xrange(opts.batch_size):
            single_sample  = file_reader.next()
            if single_sample is None:
                break
            target = single_sample[0]
            temp = single_sample[1:opts.sequence_length]
            if len(temp) < opts.sequence_length:
                gap = opts.sequence_length - len(temp)
                temp = np.array(temp + [0] * gap)
            assert len(temp) == opts.sequence_length
            batch[i] = temp
            labels[i] = target
        if len(labels) == opts.batch_size and single_sample is not None:
            yield np.array(batch), labels
        else:
            break
            
def get_substitute_cate(sample, target_index, opts):
    field_i = opts.fields_index_inverse.get(sample[target_index])
    if field_i is None:
        field_i = np.random.choice(opts.fields_index.keys(),1)[0]
    field_cates = opts.fields_index[field_i]
    rst = np.random.choice(field_cates,1)[0]
    if len(field_cates) == 1:
        rst = np.random.randint(opts.vocabulary_size)
    return rst

def generate_fake_sample(temp, opts):
        temp_sequence_length = len(temp)
        temp = temp[0:opts.sequence_length]
        if len(temp) < opts.sequence_length:
            gap = opts.sequence_length - len(temp)
            temp = np.array(temp + [0] * gap)
        else:
            temp_sequence_length = opts.sequence_length
        assert len(temp) == opts.sequence_length
        targets_to_avoid = set(temp)
        indices_to_avoid = set()
        substitute_index = np.random.randint(temp_sequence_length)
        substitute_target = get_substitute_cate(temp, substitute_index, opts)
        for _ in range(opts.substitute_num):
            while substitute_index in indices_to_avoid:
                substitute_index = np.random.randint(temp_sequence_length)
            indices_to_avoid.add(substitute_index)

            count = 0
            while substitute_target in targets_to_avoid:
                if count > 5:
                    break
                substitute_target = get_substitute_cate(temp, substitute_index, opts)
                count += 1
            targets_to_avoid.add(substitute_target)
            temp[substitute_index] = substitute_target
        return temp


def generate_discriminant_batch(opts, is_train=True, rate=0.5):
    data_index = 0
    if is_train:
        file_reader = read_file(opts.train_path)
    else:
        file_reader = read_file(opts.test_path)
    while True:
        batch = np.ndarray(shape=(opts.batch_size, opts.sequence_length))
        labels = []
        for i in xrange(opts.batch_size):
            if np.random.random() > rate:
                single_sample  = file_reader.next()
                temp = single_sample[1:opts.sequence_length]
                if len(temp) < opts.sequence_length:
                    gap = opts.sequence_length - len(temp)
                    temp = np.array(temp + [0] * gap)
                assert len(temp) == opts.sequence_length
                batch[i] = temp
                labels.append(1.)
            else:
                single_sample  = file_reader.next()
                temp = single_sample[1:opts.sequence_length]
                batch[i] = generate_fake_sample(temp, opts)
                labels.append(0.)
        yield batch, np.array(labels)
        
def read_feat_index(opts):
    vocabulary_size = 0
    reverse_dictionary_raw = np.array(pd.read_csv(opts.featindex, sep='\t', header=None))
    reverse_dictionary = {}
    dictionary = {}
    for item in reverse_dictionary_raw:
        reverse_dictionary[int(item[1])] = item[0]
        dictionary[item[0]] = int(item[1])
    if item[1] > vocabulary_size:
        vocabulary_size = item[1]
    vocabulary_size = len(dictionary.keys())
    print('vocabulary_size: ',vocabulary_size)
    return reverse_dictionary, dictionary, vocabulary_size


def eval_auc(model, opts, target=None, get_prob=None):
    testing_batch_generator = ctr_batch_generator(opts,train=False)
    batch_num = 0
    y = []
    pred = []
    for batch, labels in testing_batch_generator:
        if target is None or get_prob is None:
            probs = model.predict_proba(batch, batch_size=opts.batch_size, verbose=0)
        else:
            probs = get_prob([batch])[0]
        y.extend(labels)
        pred.extend([p[0] for p in probs])
        batch_num += 1
    fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    loss = metrics.log_loss(y, pred)
    print("Total testing sample: ", len(y), " Positive sample: ", sum(y))
    opts.auc = auc
    opts.loss = loss
    with open(opts.log_path, 'a') as f:
        f.write(str(opts.__dict__)+'\r')
    print("AUC:", auc, ', log loss: ', loss)
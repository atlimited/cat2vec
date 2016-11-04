#!/usr/bin/python
import sys
import numpy as np

if len(sys.argv) < 1:
    print 'Usage: train.index.txt'
    exit(-1)

print("Current file: ", sys.argv[1])
fi = open(sys.argv[1], 'r')
labels = []
for line in fi:
    labels.append(int(line.split(' ')[0]))
fi.close()

labels = np.array(labels)
rates = [1, 2, 5, 10, 100]
pos = np.where(labels == 1)[0]
neg = np.where(labels == 0)[0]
print('negative num: ', len(neg), "positive number: ", len(pos))
for rate in rates:
    print("making downsampling file for rate(neg:pos) ", rate)
    downsampling_num = int(rate * len(pos))
    if downsampling_num > len(neg):
        # downsampling_num = len(neg)
        print("rate is too large")
        break
    neg_balanced = np.random.choice(neg, downsampling_num, replace=False)
    balanced_indices = set(np.concatenate((pos, neg_balanced), axis=0))
    index = 0
    downsampling_file_name = sys.argv[1][
        0:sys.argv[1].rindex('.') + 1] + str(rate) + '.txt'
    fi = open(sys.argv[1], 'r')
    fo = open(downsampling_file_name, 'w')
    for line in fi:
        # label = int(line.split(' ')[0])
        if index in balanced_indices:
            fo.write(line)
        index += 1
    fi.close()
    fo.close()

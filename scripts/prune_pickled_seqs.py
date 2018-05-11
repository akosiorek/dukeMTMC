#!/usr/bin/env python2
# coding: utf-8

# In[8]:

import os
import sys
import cPickle as pickle
import numpy as np

from absl import flags


flags.DEFINE_string('input_path', 'processed/camera2_240', 'Directory with pickle files')
flags.DEFINE_string('output_path', '', 'Output directory; default to the same as `input_path`.')
flags.DEFINE_string('output_name', 'merged.pickle', 'Name of the output pickle file')
flags.DEFINE_boolean('save_pruned_seqs', True, 'Saves pruned sequences if True')


def load_pickle(path):
    with open(path, 'r') as f:
        return pickle.load(f)


def pickle_to_file(path, obj):
    with open(path, 'w') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def merge_pickles(data_dicts, output_path):
    data = data_dicts[0]
    data = {k: [v] for k, v in data.iteritems()}

    for d in data_dicts[1:]:
        for k, v in d.iteritems():
            data[k].append(v)

    print 'data:'
    for k, v in data.iteritems():
        print k
        for vv in v:
            print '\t', vv.shape

    data = {k: np.concatenate(v, 1) for k, v in data.iteritems()}
    idx = np.random.permutation(data['imgs'].shape[1])
    data = {k: v[:, idx] for k, v in data.iteritems()}

    print 'merged data:'
    for k, v in data.iteritems():
        print '\t key: {}, shape: {}'.format(k, v.shape)
        print '\t min: {:.03f}, mean: {:.03f}, max: {:.03f}, std: {:.03f}'.format(v.min(), v.mean(), v.max(), v.std())

    pickle_to_file(output_path, data)


if __name__ == '__main__':

    F = flags.FLAGS
    F(sys.argv)

    input_dir = F.input_path if os.path.isdir(F.input_path) else os.path.dirname(F.input_path)
    output_path = F.output_path if F.output_path else input_dir

    data_folders = []
    for f in os.listdir(input_dir):
        f = os.path.join(input_dir, f)
        if os.path.isdir(f) and os.path.exists(f + '.pickle'):
            data_folders.append(f)


    pruned_seqs = []
    if F.save_pruned_seqs:
        pruned_path = os.path.join(output_path, 'pruned')
        if not os.path.exists(pruned_path):
            os.mkdir(pruned_path)

    for f in data_folders:
        pickle_name = f + '.pickle'
        pruned_name = os.path.join(pruned_path, os.path.basename(pickle_name))
        data = load_pickle(pickle_name)

        seqs_to_retain = [int(i.split('.')[0]) for i in os.listdir(f) if i.endswith('.jpeg')]
        data = {k: v[:, seqs_to_retain] for k, v in data.iteritems()}

        if F.save_pruned_seqs:
            pickle_to_file(pruned_name, data)

        pruned_seqs.append(data)

    merged_path = os.path.join(pruned_path, 'merged.pickle')
    merge_pickles(pruned_seqs, merged_path)


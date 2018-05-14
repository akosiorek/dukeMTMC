#!/usr/bin/env python2
# coding: utf-8

# In[8]:

import os
import sys
import cPickle as pickle
import numpy as np

from absl import flags
import zipfile


flags.DEFINE_string('pickle_path', 'processed/camera2_240', 'Directory with pickle files')
flags.DEFINE_string('pruned_folder_path', 'processed/camera2_240', 'Directory with folders of pruned sequences')

flags.DEFINE_string('output_path', '', 'Output directory; default to the same as `pickle_path`.')
flags.DEFINE_string('output_name', 'merged.pickle', 'Name of the output pickle file')
flags.DEFINE_boolean('save_pruned_seqs', False, 'Saves pruned sequences if True')


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

    output_path = F.output_path if F.output_path else F.pickle_path

    data_folders = set()
    pickles = set()

    pickles = set(filter(lambda f: f.endswith('.pickle'), os.listdir(F.pickle_path)))
    seq_dir_to_pickle = {}

    for f in os.listdir(F.pruned_folder_path):

        basename = os.path.splitext(f)[0]
        corresponding_pickle = basename + '.pickle'
        if corresponding_pickle in pickles:
            seq_dir_to_pickle[os.path.join(F.pruned_folder_path, f)] = os.path.join(F.pickle_path, corresponding_pickle)

    pruned_seqs = []
    if F.save_pruned_seqs:
        pruned_path = os.path.join(output_path, 'pruned')
        if not os.path.exists(pruned_path):
            os.mkdir(pruned_path)

    for data_folder, pickle_path in seq_dir_to_pickle.iteritems():
        data = load_pickle(pickle_path)

        if os.path.isdir(data_folder):
            files = os.listdir(f)
        elif data_folder.endswith('.zip'):
            files = map(os.path.basename, zipfile.ZipFile(data_folder).namelist())

        seqs_to_retain = [int(i.split('.')[0]) for i in files if i.endswith('.jpeg')]
        data = {k: v[:, seqs_to_retain] for k, v in data.iteritems()}

        if F.save_pruned_seqs:
            pickle_to_file(pruned_name, data)

        pruned_seqs.append(data)

    merged_path = F.output_path
    if not merged_path.endswith('.pickle'):
        merged_path = os.path.join(merged_path, F.output_name)
    merge_pickles(pruned_seqs, merged_path)


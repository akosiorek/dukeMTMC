#!/usr/bin/env python3

import os
import sys
import numpy as np

if __name__ == '__main__':

    path = sys.argv[1]
    dirs = [os.path.join(path, f) for f in os.listdir(path)]
    lens = [len(os.listdir(d)) for d in dirs]

    print('Created {} sequences.'.format(len(lens)))
    print('\tMin length: {}'.format(min(lens)))
    print('\tMax length: {}'.format(max(lens)))
    print('\tMean length: {}'.format(np.mean(lens)))
    print('\tMedian length: {}'.format(np.median(lens)))
#!/usr/bin/env python3

import os
import shutil
import sys


def prune(input_dir, min_seq_len):

    dirs = os.listdir(input_dir)

    for d in dirs:
        d = os.path.join(input_dir, d)
        n = len([f for f in os.listdir(d) if f.endswith('jpeg')])
        if n < min_seq_len:
            print('Removing {} with {} images'.format(d, n))
            shutil.rmtree(d)

if __name__ == '__main__':

    args = sys.argv[1:]
    src = args[0]
    min_seq_len = int(args[1])
    prune(src, min_seq_len)
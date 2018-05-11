#!/usr/bin/env python2
import sys

from prune_pickled_seqs import load_pickle, merge_pickles


if __name__ == '__main__':

    args = sys.argv[1:]
    inpt_paths = args[:-1]
    output_path = args[-1]

    pickles = [load_pickle(p) for p in inpt_paths]
    merge_pickles(pickles, output_path)
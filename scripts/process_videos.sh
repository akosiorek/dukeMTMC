#!/usr/bin/env sh

input_folder=$1
results_folder=$2

every_nth_frame=30

# subtract background and split into images
./scripts/subtract_background.py $input_folder $results_folder $every_nth_frame

# Split into sequences (and crop the inner 50% if a fraction of .5 is provided as the last argument)
./scripts/split.py $results_folder/frames $results_folder/seqs

# Prune sequences that are shorter than three time-steps
./scripts/prune.py $results_folder/seqs 3

# Print a summary
./scripts/summarise.py $results_folder/seqs
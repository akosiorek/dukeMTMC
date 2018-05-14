#!/usr/bin/env sh

./scripts/prune_pickled_seqs.py --pickle_path processed/camera1_240_gmm_30 --pruned_folder_path processed/gdrive/done/cam1_bz2_done --output_path processed/gdrive/cam1.pickle
./scripts/prune_pickled_seqs.py --pickle_path processed/camera5_240_gmm --pruned_folder_path processed/gdrive/done/cam5_bz2_done --output_path processed/gdrive/cam5.pickle
./scripts/prune_pickled_seqs.py --pickle_path processed/camera7_240_gmm --pruned_folder_path processed/gdrive/done/cam7_bz2_done --output_path processed/gdrive/cam7.pickle

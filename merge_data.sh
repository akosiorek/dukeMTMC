#!/usr/bin/env sh

./scripts/merge_pickles.py\
    processed/camera2_240/pruned_duke_cam2_240_merged.pickle\
    processed/camera2_240/pruned/merged.pickle\
    processed/camera5_240_gmm/pruned/merged.pickle\
    processed/gdrive/cam1.pickle\
    processed/gdrive/cam5.pickle\
    processed/gdrive/cam7.pickle\
        processed/gdrive/merged.pickle
    

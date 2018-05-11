
# coding: utf-8

# In[1]:

import os
import sys
import shutil
import cPickle as pickle
from attrdict import AttrDict

import numpy as np

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

import cv2

from tensorflow.python.util import nest

from absl import flags

flags.DEFINE_integer('cam', 2, '')


# indices to columns of the ground-truth matrix
key = 'camera ID frame left top width height worldX worldY feetX feetyY'.split()
key = {k: v for v, k in enumerate(key)}

# cameras are synshcornised with respect to camera #5 and these are frame offsets
start_frame_nums = [5543, 3607, 27244, 31182, 1, 22402, 18968, 46766]

# the dataset will be composed of sequences of this length
target_sequence_length = 5

# num pixels in the longer side of the original videos
original_longer = 1920

# relative path to the folder with all frames
frames_folder = 'frames'


if __name__ == '__main__':
    F = flags.FLAGS
    F(sys.argv)

    # path to the ground-truth file
    gt_path = '../trainvalRaw.mat'
    # path = '../trainval.mat'

    # path to the folder with extracted frames
    data_path = '../processed/camera{}_240'.format(F.cam)


    # num pixels in the longer side of downscaled videos
    downscaled_longer = 240

    roi_size = 64, 64
    roi = None


    # the final dataset will be written to this file
    pickle_path = 'duke_cam_{}_240_roi_{}_t_5.pickle'

    max_n_objects = 3

    # fraction of object that needs to be within the roi to be counted as present
    intersection_threshold = .25
    assert 0. <= intersection_threshold <= 1.

    # maximum allowed number of consecutive empty frames in a sequence
    max_empty_frames = 1

    # minimum sequence length
    min_seq_len = 5


    # In[4]:

    if roi is None:
        u = np.random.uniform(size=2)
        bounds = np.asarray([downscaled_longer * 9./16, downscaled_longer]) - roi_size
        yx = np.round(bounds * u).astype(np.int32)
        roi = tuple(yx) + roi_size
        print 'roi =', roi

    ratio = float(original_longer) / downscaled_longer
    original_roi = ratio * np.asarray(roi)
    start_frame_num = start_frame_nums[F.cam - 1]

    frames_folder = os.path.join(data_path, frames_folder)
    pickle_path = os.path.join(data_path, pickle_path.format(F.cam, '_'.join([str(s) for s in roi])))


    # In[5]:

    print 'Loading "{}"'.format(os.path.basename(gt_path))

    if 'Raw' in gt_path:
        import h5py
        f = h5py.File(gt_path, 'r')
        data = f['trainData'][()].T
    else:
        import scipy.io
        f = scipy.io.loadmat(gt_path)
        data = f['trainData']


    # In[6]:

    # Select data for the specified camera and shift frame nums by the starting frame num
    camera_idx = np.equal(data[:, 0], F.cam)
    ids, frames, left, top, width, height = np.split(data[camera_idx, 1:7], 6, -1)

    boxes = np.concatenate([top, left, height, width], -1)
    # frames -= start_frame_num

    objects = np.concatenate([frames, ids, boxes], -1) # frame, bbox

    print 'Total number of objects:', objects.shape[0]
    print 'Frames from {} to {}'.format(frames.min(), frames.max())


    # In[7]:

    # Select objects that are only withing the region of interest
    def intersection(bbox, roi):
        """Computes area of intersection between boxes in `bbox` and boxes in `roi`.
        Dimensions in both should be given as (y, x, h, w).

        :param bbox: np.array of shape [N, 4]
        :param roi: np.array of shape [4] or [k, 4] where k \in {1, N}
        :return: np.array of shape [N]
        """

        bbox = np.asarray(bbox)
        roi = np.asarray(roi)

        while len(roi.shape) < len(bbox.shape):
            roi = roi[np.newaxis, ...]

        y_top = np.maximum(bbox[..., 0], roi[..., 0])
        x_left = np.maximum(bbox[..., 1], roi[..., 1])
        y_bottom = np.minimum(bbox[..., 0] + bbox[..., 2], roi[..., 0] + roi[..., 2])
        x_right = np.minimum(bbox[..., 1] + bbox[..., 3], roi[..., 1] + roi[..., 3])

        invalid_x = np.less(x_right, x_left)
        invalid_y = np.less(y_bottom, y_top)
        invalid = np.logical_or(invalid_x, invalid_y)

        # The intersection of two axis-aligned bounding boxes is always an
        # axis-aligned bounding box
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        intersection_area[invalid] = 0.
        return intersection_area


    def fraction_of_object_area_in_roi(bbox, roi):
        intersection_area = intersection(bbox, roi)
        object_area = np.prod(bbox[..., 2:], -1)
        return intersection_area / (object_area + 1e-8)

    object_fraction_within_roi = fraction_of_object_area_in_roi(objects[..., 2:], original_roi)
    is_within_roi = np.greater(object_fraction_within_roi, intersection_threshold)

    objects = np.concatenate((objects, object_fraction_within_roi[:, np.newaxis]), -1)
    objects = objects[is_within_roi]

    print 'Total number of objects within ROI:', objects.shape[0]

    # Count objects in all remaining frames frames
    unique_frames, object_counts = np.unique(objects[..., 0], return_counts=True)


    # In[ ]:

    # Find all frames we have
    def find_frames(path):
        frames_paths = [os.path.join(path, p) for p in os.listdir(path) if p.endswith('jpeg')]
        frames = {int(p.split('_')[-1].split('.')[0]): p for p in frames_paths}
        return frames

    frames = find_frames(frames_folder)
    frame_nums = sorted(frames.keys())


    # In[ ]:

    frame_info = dict()

    object_counts = {k: v for k, v in zip(unique_frames, object_counts)}

    for frame_num, path in frames.iteritems():
        frame_info[frame_num] = AttrDict()
        if frame_num in object_counts:
            count = object_counts[frame_num]
        else:
            count = 0
            
        frame_info[frame_num]['count'] = count
        frame_info[frame_num]['path'] = path
        
        this_frame_idx = np.equal(objects[:, 0], frame_num)
        this_frame_info = objects[this_frame_idx, 1:]
        
        frame_info[frame_num]['ids'] = this_frame_info[:, 0]
        frame_info[frame_num]['boxes'] = this_frame_info[:, 1:-1]
        frame_info[frame_num]['frac_within_roi'] = this_frame_info[:, -1]


    # In[ ]:

    frame_info[frame_num - 300]


    # In[ ]:

    # creates sequences
    seqs = [] # a list of list of consecutive frame numbers

    last_frame_was_empty = None
    n_empty_frames = 0
    seq = []

    frame_nums = sorted(frame_info.keys())

    # for i, (frame_num, obj_count) in enumerate(regularly_spaced_frames_and_counts):
    for frame_num in frame_nums:
        
    #     frame_is_invalid = (obj_count == -1)
    #     frame_is_empty = (obj_count == 0)
        
        frame_is_invalid = (frame_info[frame_num].count > max_n_objects)
        frame_is_empty = (frame_info[frame_num].count == 0) or (frame_info[frame_num].frac_within_roi.sum() < 1.)
        
        if frame_is_invalid:
            n_empty_frames = 0
                
            if len(seq) > 0:
                seqs.append(seq)
                seq = []
           
        elif frame_is_empty and len(seq) > 0:
            if n_empty_frames < max_empty_frames:
                n_empty_frames += 1
                seq.append(frame_num)
            else:
                n_empty_frames = 0
                seqs.append(seq)
                seq = []
                
        else:
            if len(seq) == 0 and not frame_is_empty:
                if last_frame_was_empty:
                    seq.append(last_frame)
                    
            seq.append(frame_num)
            
        last_frame = frame_num
        last_frame_was_empty = frame_is_empty
        last_frame_was_valid = not frame_is_invalid


    # In[ ]:

    import itertools
    all_valid_frames = list(itertools.chain(*seqs))
    print 'Number of valid frames', len(all_valid_frames)


    # In[ ]:

    # Splits sequences into fixed-length ones
    fixed_length_seqs = []
    for seq in seqs:
        if len(seq) < target_sequence_length:
            continue
        
        for t in xrange(0, len(seq) + 1, target_sequence_length):
            fixed_length_seqs.append(seq[t:t+target_sequence_length])
            
        if len(fixed_length_seqs[-1]) < target_sequence_length // 2:
            del fixed_length_seqs[-1]

    seqs = fixed_length_seqs


    # In[ ]:

    # # prune sequences
    for i in xrange(len(seqs) - 1, -1, -1):
        if len(seqs[i]) < min_seq_len:
            del seqs[i]


    # In[ ]:

    seq_lens = [len(seq) for seq in seqs]
    print 'Created {} sequences:'.format(len(seqs))
    print '\tmin seq_len =', min(seq_lens)
    print '\tmean seq_len =', np.mean(seq_lens)
    print '\tmedian seq_len =', np.median(seq_lens)
    print '\tmax seq_len =', max(seq_lens)
    print '\ttotal number of frames =', sum(seq_lens)


    # In[ ]:

    # for every frame, figure out how many object are there
    flat_seqs = nest.flatten(seqs)
    flat_nums = [frame_info[i].count for i in flat_seqs]
    nums = nest.pack_sequence_as(seqs, flat_nums)
    print 'Average number of objects per frame =', np.mean(nums)


    # In[ ]:

    # Create datasets
    y, x, h, w = roi     

    for i, seq in enumerate(seqs):
        for j, frame_num in enumerate(seq):

            img = cv2.imread(frame_info[frame_num].path)
            img = img[y:y+h, x:x+w]
            seq[j] = img
            
        seqs[i] = np.asarray(seq)
        nums[i] = np.asarray(nums[i])
            
    seqs = np.stack(seqs, 1)
    nums = np.stack(nums, 1).astype(np.int32)[..., np.newaxis]
    print seqs.shape, nums.shape


    # In[ ]:

    def pickle_to_file(path, obj):
        with open(path, 'w') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    data = dict(
        imgs=seqs,
        nums=nums
    )

    pickle_to_file(pickle_path, data)


    # In[ ]:

    seq_img_folder = os.path.splitext(pickle_path)[0]
    if not os.path.exists(seq_img_folder):
        os.mkdir(seq_img_folder)


    seqs = data['imgs']
    n_timesteps = seqs.shape[0]
    n_seqs = seqs.shape[1]

    scale = 2.
    for seq_num in xrange(n_seqs):
        print '{}/{}\r'.format(seq_num+1, n_seqs),
        sys.stdout.flush()
        
        fig, ax = plt.subplots(1, n_timesteps, figsize=scale*np.asarray([n_timesteps, 1]), sharex=True, sharey=True)
        ax[0].set_ylabel(seq_num)
        for t in xrange(n_timesteps):
            ax[t].imshow(seqs[t, seq_num])
            
        figpath = os.path.join(seq_img_folder, '{:04d}.jpeg'.format(seq_num))
        fig.savefig(figpath, dpi=77, bbox_inches='tight')
        plt.close(fig)
        
    print

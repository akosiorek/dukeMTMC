#!/usr/bin/env python3

import os
import sys
import numpy as np
import cv2

from absl import flags

flags.DEFINE_string('input_path', '', '')
flags.DEFINE_string('output_path', '', '')
flags.DEFINE_integer('every_nth_frame', 1, '')
flags.DEFINE_integer('output_width', 0, '')


class AbstractBackgroundModel(object):

    def __call__(self, frame):
        fg_mask = self._fg_mask(frame)

        fg_mask = (fg_mask / 255).astype(np.float32)
        frame = frame.astype(np.float32) / 255
        frame *= fg_mask[..., np.newaxis]
        # frame = (1. - frame) * fg_mask[..., np.newaxis]
        # frame = np.round(frame * 255).astype(np.uint8)
        return frame


class BackgroundModelGMM(AbstractBackgroundModel):

    def __init__(self):
        self._model = cv2.bgsegm.createBackgroundSubtractorMOG(history=1000, nmixtures=10, backgroundRatio=.7)

    def _fg_mask(self, frame):
        return self._model.apply(frame)


class BackgroundModelGMG(AbstractBackgroundModel):

    def __init__(self):
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self._model = cv2.bgsegm.createBackgroundSubtractorGMG(initializationFrames=1000, decisionThreshold=.9)

    def _fg_mask(self, frame):
        fg_mask = self._model.apply(frame)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self._kernel)
        return fg_mask


def create_output_video(output_path, frame_num, frame, codec='XVID', fps=30):
    shape = frame.shape
    is_color = len(frame.shape) == 3 and frame.shape[-1] == 3
    frame_size = shape[:2]
    print(frame_size)
    frame_size = frame_size[1], frame_size[0]

    fourcc = cv2.VideoWriter_fourcc(*codec)

    path, ext = os.path.splitext(output_path)
    if not ext:
        ext = '.avi'

    base_name = os.path.basename(path)
    name = '{}_{}{}'.format(base_name, frame_num, ext)
    output_path = os.path.join(os.path.join(path, 'videos'), name)

    output_video = cv2.VideoWriter(output_path, fourcc, fps, frame_size,
                                   isColor=is_color)
    return output_video


def subtract_background(background_subtraction, input_path, output_path='', starting_frame_num=0,
                        every_nth_frame=1, output_width=0):

    print('Processing', input_path)
    video_name = os.path.basename(input_path)

    input_video = cv2.VideoCapture(input_path)
    fps = input_video.get(cv2.CAP_PROP_FPS)
    total_num_frames = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT)) + starting_frame_num

    output_folder = None
    if output_path:
        output_folder = os.path.splitext(output_path)[0]
        img_path = os.path.join(output_folder, 'frames/img_{:06d}.jpeg')
        vid_folder = os.path.join(output_folder, 'frames')
        frame_folder = os.path.join(output_folder, 'videos')
        folders = [output_folder, vid_folder, frame_folder]
        for f in folders:
            if not os.path.exists(f):
                os.mkdir(f)

    print('output_path', output_path)
    frame_num = starting_frame_num
    while True:
        frame_num += 1
        print('Processing frame #{}/{}'.format(frame_num, total_num_frames), end='\r')
        sys.stdout.flush()

        ret, frame = input_video.read()
        if ret is False:
            break

        if frame_num == starting_frame_num+1 and output_path is not None:
            output_video = create_output_video(output_path, frame_num, frame, fps=fps)

        frame = cv2.bilateralFilter(frame, 25, 15, 15)
        frame = background_subtraction(frame)

        if output_width != 0:
            height, width = frame.shape[:2]
            ratio = float(output_width) / width
            output_height = np.round(height * ratio).astype(np.int32)
            frame = cv2.resize(frame, (output_width, output_height))

        if output_path:
            output_video.write(frame)

            if frame_num % every_nth_frame == 0:
                cv2.imwrite(img_path.format(frame_num), frame)
        else:

            cv2.imshow(video_name, frame)
            cv2.waitKey(int(1000. / fps) // 2)

    print()

    input_video.release()
    if output_path:
        output_video.release()

    cv2.destroyAllWindows()

    return frame_num - 1

if __name__ == '__main__':

    F = flags.FLAGS
    F(sys.argv)

    background_subtraction = BackgroundModelGMM()
    # background_subtraction = BackgroundModelGMG()

    if os.path.isdir(F.input_path):
        videos = [f for f in os.listdir(F.input_path) if f.endswith('avi')]

        def num(filename):
            return int(os.path.splitext(filename)[0])

        videos = sorted(videos, key=num)
        videos = [os.path.join(F.input_path, f) for f in videos]
    else:
        videos = [F.input_path]

    starting_frame = 0
    for video in videos:
        starting_frame = subtract_background(background_subtraction, video, F.output_path, starting_frame,
                                             F.every_nth_frame, F.output_width)

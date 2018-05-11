#!/usr/bin/env python3

import os
import sys
import numpy as np
import cv2

import bgsubcnt

from absl import flags

flags.DEFINE_string('input_path', '', 'Directory or a path to an *.avi files; If directory, all '
                                      'avi files in this directory will be processed.')
flags.DEFINE_string('output_path', '', 'Directory where all results will be stored.')
flags.DEFINE_integer('every_nth_frame', 1, 'If greater than zero, then `every_nth_frame` - 1 '
                                           'frames are skipped between frames.')
flags.DEFINE_integer('output_width', 0, 'Rescales images and videos to this width if != 0.')
flags.DEFINE_string('mode', 'gmm', 'Choose from {gmm, gmg, cnt}.')


class AbstractBackgroundModel(object):

    def __call__(self, frame):
        fg_mask = self._fg_mask(frame)
        fg_mask = (fg_mask / 255).astype(np.float32)
        frame = frame.astype(np.float32) / 255

        # nnz = fg_mask[np.not_equal(fg_mask, 0.)]
        # if np.count_nonzero(nnz) > 0:
        #     print('min = {:.03f}, mean = {:.03f}, max = {:.03f}'.format(nnz.min(), nnz.mean(), nnz.max()))
        frame *= fg_mask[..., np.newaxis]
        return frame


class BackgroundModelGMM(AbstractBackgroundModel):

    def __init__(self):
        self._model = cv2.bgsegm.createBackgroundSubtractorMOG(history=500, nmixtures=10, backgroundRatio=.7)
        self._erosion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        self._dialation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))

    def _fg_mask(self, frame):
        frame = cv2.medianBlur(frame, 5)
        fg = self._model.apply(frame)

        cv2.erode(fg, self._erosion_kernel, fg)
        cv2.dilate(fg, self._dialation_kernel, fg, iterations=2)
        return fg


class BackgroundModelGMG(AbstractBackgroundModel):

    def __init__(self):
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self._model = cv2.bgsegm.createBackgroundSubtractorGMG(initializationFrames=120, decisionThreshold=.9)

    def _fg_mask(self, frame):
        fg_mask = self._model.apply(frame)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self._kernel)
        return fg_mask


class BackgroundModelCNT(AbstractBackgroundModel):

    def __init__(self):
        self._model = bgsubcnt.createBackgroundSubtractor(5, True, 5*60)

    def _fg_mask(self, frame):
        # frame = cv2.bilateralFilter(frame, 9, 150, 150)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.medianBlur(frame, 3)
        return self._model.apply(frame)


def equalize_histogram(img):

    rgb = np.split(img, 3, -1)
    for i, c in enumerate(rgb):
        rgb[i] = cv2.equalizeHist(c.squeeze())

    return np.stack(rgb, -1)


def create_output_video(output_path, frame_num, frame, codec='XVID', fps=30):
    shape = frame.shape
    is_color = len(frame.shape) == 3 and frame.shape[-1] == 3
    frame_size = shape[:2]
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


    frame_num = starting_frame_num
    while True:
        frame_num += 1
        print('Processing frame #{}/{}'.format(frame_num, total_num_frames), end='\r')
        sys.stdout.flush()

        ret, frame = input_video.read()
        if ret is False:
            break



        frame = 255 - frame
        # frame = equalize_histogram(frame)
        # frame = cv2.bilateralFilter(frame, 25, 15, 15)
        fg_frame = background_subtraction(frame)
        fg_frame = cv2.medianBlur(fg_frame, 3)
        # frame = cv2.bilateralFilter(frame, 5, 15, 15)

        idx = np.greater(fg_frame, 50)
        fg_frame[idx] = frame[idx]
        frame = fg_frame

        frame = frame.astype(np.float32)
        frame *= 255. / (frame.max() + 1e-8)
        frame = np.round(frame).astype(np.uint8)

        if output_width != 0:
            height, width = frame.shape[:2]
            ratio = float(output_width) / width
            output_height = np.round(height * ratio).astype(np.int32)
            frame = cv2.resize(frame, (output_width, output_height))

        # print(frame.shape, frame.dtype, frame.max())
        # print()
        if output_path:
            if frame_num == starting_frame_num + 1:
                output_video = create_output_video(output_path, frame_num, frame, fps=fps)

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

    if F.mode == 'gmm':
        background_subtraction = BackgroundModelGMM()
    elif F.mode == 'gmg':
        background_subtraction = BackgroundModelGMG()
    elif F.mode == 'cnt':
        background_subtraction = BackgroundModelCNT()
    else:
        print('Invalid mode = "{}"'.format(F.mode))
        sys.exit(1)

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

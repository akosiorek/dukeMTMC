#!/usr/bin/env python3

import os
import sys
import cv2
import numpy as np


def find_imgs(path):
    img_paths = []
    for p in os.listdir(path):
        if not p.endswith('jpeg'):
            continue

        num = int(os.path.splitext(p)[0].split('_')[-1])
        img_paths.append((num, p))
    img_paths = sorted(img_paths, key=lambda x: x[0])
    img_paths = [os.path.join(path, p[1]) for p in img_paths]
    return img_paths


def output_img_path(output_dir, seq_num, img_path):
    img_name = os.path.basename(img_path)
    output_dir = os.path.join(output_dir, str(seq_num))
    return os.path.join(output_dir, img_name)


def maybe_create_seq_folder(output_dir, seq_num):
    seq_dir = os.path.join(output_dir, str(seq_num))
    if not os.path.exists(seq_dir):
        os.makedirs(seq_dir)


def write_result_img(output_dir, seq_num, img_path, img):
    maybe_create_seq_folder(output_dir, seq_num)
    img_path = output_img_path(output_dir, seq_num, img_path)
    print('writing', img_path)
    cv2.imwrite(img_path, img)

def crop(img, fraction):
    img_size = np.asarray(img.shape[:2])
    centre = img_size // 2
    radius = np.round(centre * fraction).astype(np.int32)

    return img[centre[0]-radius[0]:centre[0]+radius[0], centre[1]-radius[1]:centre[1]+radius[1]]


def split(input_dir, output_dir, fraction):

    img_paths = find_imgs(input_dir)
    seq_num = 0
    all_zeros = False
    last_img = None
    last_path = None

    maybe_create_seq_folder(output_dir, seq_num)

    skipped = 0
    for img_path in img_paths:
        img = cv2.imread(img_path)
        img = crop(img, fraction)
        # img = cv2.medianBlur(img, 3)

        if img.mean() > 1.4:
            skipped += 1
            continue
        else:
            if skipped > 0:
                print('skipped {} frames'.format(skipped))
                if skipped > 2:
                    seq_num += 1
                    maybe_create_seq_folder(output_dir, seq_num)

                skipped = 0

        if img.mean() < 0.01:
            if all_zeros:
                continue
            else:
                write_result_img(output_dir, seq_num, img_path, img)
                all_zeros = True
                seq_num += 1
        else:
            if all_zeros:
                all_zeros = False
                write_result_img(output_dir, seq_num, past_path, last_img)

            write_result_img(output_dir, seq_num, img_path, img)

        last_img, past_path = img, img_path


if __name__ == '__main__':

    args = sys.argv[1:]
    src = args[0]
    dst = args[1]
    fraction = float(args[2]) if len(args) > 2 else 1.

    split(src, dst, fraction)

#!/usr/bin/env python3

import os
import shutil
import sys

import fnmatch
import cv2


def crop(input_dir, output_dir, y, x, h, w):

    pattern = '*.jpeg'
    img_paths = [f for f in os.listdir(input_dir) if fnmatch.fnmatch(f, pattern)]

    for img_path in img_paths:
        print('Processing', img_path)

        input_path = os.path.join(input_dir, img_path)
        output_path = os.path.join(output_dir, img_path)
        output_folder = os.path.dirname(output_path)

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        img = cv2.imread(input_path)
        img = img[y:y+h, x:x+w]
        cv2.imwrite(output_path, img)

if __name__ == '__main__':

    args = sys.argv[1:]
    # print(args)
    src = args[0]
    dst = args[1]
    dims = [int(i) for i in args[2:6]]
    crop(src, dst, *dims)

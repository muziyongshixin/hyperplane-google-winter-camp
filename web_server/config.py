import os
import collections


def listdir_with_prefix(dir_, prefix):
    paths = []
    for fname in os.listdir(dir_):
        if fname.startswith(prefix):
            paths.append((fname, os.path.join(dir_, fname)))
    return paths


CROP_SCALE = 1

RESULT_BASEDIR = 'results'

VIDEO_BASEDIR = 'videos'

DATABASEDIR = "F:/copy_detection_dataset/VCDB/core_dataset/core_dataset"
# DATABASEDIR = "/scratch/zhuangn/VCDB/core_dataset/core_dataset"
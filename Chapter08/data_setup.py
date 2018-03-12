# Deep Learning Quick Reference Chapter 8: Transfer Learning
# Mike Bernico <mike.bernico@gmail.com>
# This data setup script expects the kaggle dog vs. cat dataset present in a folder called train
# located under the chapter_8 directory
# Kaggle Dog Vs. Cat Dataset can be found at https://www.kaggle.com/c/dogs-vs-cats/data


import os
import shutil
from pathlib import Path


def make_dir(dir):
    try:
        os.stat(dir)
    except:
        os.mkdir(dir)


def setup_dirs(dest_dir, train_dir, val_dir, test_dir):
    make_dir(dest_dir)  # data
    for folder in [train_dir, val_dir, test_dir]:
        make_dir(Path(dest_dir + folder))  # data/test
        _ = [make_dir(Path(dest_dir + folder + x)) for x in ["cat", "dog"]]  # creates cat and dog directories under folder


def copy_data(train_range, val_range, test_range, source_dir, dest_dir, train_dir, val_dir, test_dir):
    for a in ["cat", "dog"]:
        ranges = [ train_range, val_range, test_range]
        folders = [train_dir, val_dir, test_dir]
        for r, f in zip(ranges, folders):
            for i in r:
                src = Path(source_dir + a + "." + str(i) + ".jpg")
                dst = Path(dest_dir + f + a + "/" + a + "." + str(i) + ".jpg")
                shutil.copyfile(src, dst)


def main():
    dest_dir = "data/"
    source_dir = "train/"

    # append these directories to /data
    test_dir = "test/"
    val_dir = "val/"
    train_dir = "train/"
    # test and val will be 10%
    test_range = range(11249, 12500)
    val_range = range(10000, 11250)
    train_range = range(0, 10000)

    setup_dirs(dest_dir, train_dir, val_dir, test_dir)
    copy_data(train_range, val_range, test_range, source_dir, dest_dir, train_dir, val_dir, test_dir)


if __name__ == "__main__":
    main()


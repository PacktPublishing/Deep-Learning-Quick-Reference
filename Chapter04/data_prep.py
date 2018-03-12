# Deep Learning Quick Reference Chapter 4: Using  Deep Learning To Solve Binary Classification  Problems
# Mike Bernico <mike.bernico@gmail.com>
# Data prep code - Creates train/val/test split for Epileptic Seizure Recognition Data Set
# NOTE: The reader should not need to run this.

import pandas as pd
from sklearn.model_selection import train_test_split
import os
import random
random.seed = 42


def safe_create_directory(dir_name):
    if not os.path.isdir((os.path.join(os.getcwd(), "data", dir_name))):
        os.mkdir(os.path.join(os.getcwd(), "data", dir_name))


def create_data_dirs():
    [safe_create_directory(dir) for dir in ["train", "val", "test"]]


def train_val_test_split(df, val_pct=0.1, test_pct=0.1):
    size = df.shape[0]
    val_pct = (val_pct * size) / (size * (1 - test_pct))
    train_val, test = train_test_split(df, test_size=test_pct)
    train, val = train_test_split(train_val, test_size=val_pct)
    return train, val, test


def serialize_dataset(dataset):
    for k,v in dataset.items():
        out_filename = os.path.join(os.getcwd(), "data", k, k + "_data.csv")
        v.to_csv(out_filename, sep=",", index=False)
        print("Writing " + k + " to " + out_filename + " Shape:" + str(v.shape))


def main():
    df = pd.read_csv("./data/data.csv", sep=",")
    # drop the identifier row
    df = df.drop("Unnamed: 0", axis=1)
    # change classes 2-5 to 0.  1 is a seizure.  Binary Classification
    df.loc[df.y > 1, "y"] = 0
    create_data_dirs()
    dataset = dict()
    dataset["train"], dataset["val"], dataset["test"] = train_val_test_split(df)
    serialize_dataset(dataset)


if __name__ == "__main__":
    main()

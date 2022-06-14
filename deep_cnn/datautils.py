import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .logger import logger


def pp_process_input(
    root_dir: str,
    data_dir: str,
    metadata: str,
    oversample: bool,
    verbose: bool,
    perception_study: str,
):
    """
    Load place pulse image metadata and format for dataloader
    """
    images_df = create_image_df(data_dir)  # read image names
    images_df = add_qscore(
        root_dir, images_df, perception_study, metadata
    )  # get outcome labels
    images_df = scale_data(images_df, 1, 10)  # scale outcome label

    # split train, val, test
    df_train, df_val, df_test = split_meta(images_df, 0.65, 0.05, 0.3)
    if oversample is True:
        df_train = oversample_images(df_train)
    if verbose is True:
        # plot histograms
        df_train["trueskill.score_norm"].hist()
        plt.savefig(root_dir + "outputs/plots/train_dist.png")
        plt.clf()
        df_val["trueskill.score_norm"].hist()
        plt.savefig(root_dir + "outputs/plots/val_dist.png")
        plt.clf()
        df_test["trueskill.score_norm"].hist()
        plt.savefig(root_dir + "outputs/plots/test_dist.png")
        plt.clf()
        pass

    return df_train, df_val, df_test


def split_meta(images_df, train_size, val_size, test_size):
    df_ = images_df.sample(frac=train_size + val_size)
    df_val = df_.sample(frac=0.07)
    df_train = df_.drop(df_val.index)
    df_test = images_df.drop(df_train.index)
    logger.info(
        "There are %s images in the original training set" % str(df_train.shape[0])
    )
    logger.info(
        "There were %s images in the original validation set" % str(df_val.shape[0])
    )
    logger.info("There were %s images in the original test set" % str(df_test.shape[0]))
    return df_train, df_val, df_test


def get_image_id(f):
    id = []
    for i in range(len(f)):
        id.append(f[i].split("_")[2])
    return id


def create_image_df(data_dir):
    """DataFrame of image names (as found in metadata.csv)
    and location of image file"""
    files = os.listdir(data_dir)
    img_id = get_image_id(files)
    df_img = pd.DataFrame({"file": files, "location_id": img_id})
    return df_img


def add_qscore(
    root_dir,
    images_df,
    perception_study,
    metadata,
):
    """Read in metadata to add qscore label to image dataframe"""
    meta_path = str(Path(root_dir, metadata)) + "/qscores.tsv"
    meta = pd.read_csv(meta_path, sep="\t")
    perception_meta = meta[meta["study_id"] == perception_study]
    images_df.insert(
        0,
        "trueskill.score",
        images_df["location_id"].map(
            perception_meta.set_index("location_id")["trueskill.score"]
        ),
    )
    return images_df


def scale_data(images_df, start, end):
    """Scale output label into a range (start, end)"""
    width = end - start
    max_ = images_df["trueskill.score"].max()
    min_ = images_df["trueskill.score"].min()
    images_df["trueskill.score_norm"] = images_df["trueskill.score"].apply(
        lambda x: (x - min_) / (max_ - min_) * width + start
    )
    return images_df


def oversample_images(df_train):
    """Function to oversample qscores to achieve equal counts in each decile"""
    df_train["bins"] = df_train["trueskill.score_norm"].apply(lambda x: np.round(x))
    M = df_train["bins"].value_counts().max()
    frames = pd.DataFrame()
    for class_idx, group in df_train.groupby("bins"):
        oversample_class = group.sample(M - len(group), replace=True)
        frames = pd.concat([frames, oversample_class])
    df_train = pd.concat([frames, df_train])
    logger.info(
        "After oversampling, there are now %s images in the training dataset"
        % str(df_train.shape[0])
    )
    return df_train

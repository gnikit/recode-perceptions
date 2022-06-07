from deep_cnn.datautils import add_qscore, create_image_df, scale_data, split_meta


def test_image_dataframe(test_data):
    """Test images are being found"""
    test_image_df = create_image_df(data_dir=test_data)
    # =================================
    # TEST SUITE
    # =================================
    # Check the dictionary has read both images
    assert test_image_df.shape == (2, 2)


def test_adding_labels(root_dir, test_data):
    """Test outcome scores are added"""
    test_image_df = create_image_df(data_dir=test_data)
    labels_df = add_qscore(
        root_dir=root_dir,
        images_df=test_image_df,
        metadata="tests/test_input/meta/qscores.tsv",
    )
    # =================================
    # TEST SUITE
    # =================================
    # Check labels are added
    assert labels_df["trueskill.score"] is not None
    assert test_image_df.shape == (2, 3)


def test_scale_scores(root_dir, test_data):
    """Test outcome scores are added"""
    test_image_df = create_image_df(data_dir=test_data)
    labels_df = add_qscore(
        root_dir=root_dir,
        images_df=test_image_df,
        metadata="tests/test_input/meta/qscores.tsv",
    )
    images_df = scale_data(labels_df, 1, 10)
    # =================================
    # TEST SUITE
    # =================================
    # Check outcome score is in [0,10]
    assert images_df["trueskill.score_norm"].between(0, 10).all()


def test_split_meta(root_dir, test_data):
    """Test outcome scores are added"""
    test_image_df = create_image_df(data_dir=test_data)
    labels_df = add_qscore(
        root_dir=root_dir,
        images_df=test_image_df,
        metadata="tests/test_input/meta/qscores.tsv",
    )
    images_df = scale_data(labels_df, 1, 10)
    df_train, df_val, df_test = split_meta(images_df, 0.5, 0.00, 0.5)
    # =================================
    # TEST SUITE
    # =================================
    # Check train and test images are of size 1
    assert df_train.shape[0] == 1
    assert df_test.shape[0] == 1
    assert df_val.shape[0] == 0


# def test_dataset_generator():
#     """Test pre-processing in [0,1]
#     and correct image labels"""


# def test_one_training_epoch():
#     """Tests one forward and on backward pass
#     """


# df_train, df_val, df_test = pp_process_input(
#     perception_study="50a68a51fdc9f05596000002",
#     root_dir=root_dir,
#     data_dir=test_data,
#     oversample=False,
#     verbose=False,
# )

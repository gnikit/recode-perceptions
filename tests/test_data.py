from deep_cnn.datautils import add_qscore, create_image_df


def test_image_dataframe(test_data):
    """Test images are read in"""
    test_image_df = create_image_df(data_dir=test_data)
    # =================================
    # TEST SUITE
    # =================================
    # Check the dictionary has read both images
    assert test_image_df.shape == (2, 2)
    return test_image_df


def test_adding_labels(root_dir, test_data):
    """Test scores are added"""
    test_image_df = create_image_df(data_dir=test_data)
    labels_df = add_qscore(root_dir=root_dir, images_df=test_image_df)
    # =================================
    # TEST SUITE
    # =================================
    # Check labels are added
    assert labels_df["trueskill.score"] is not None


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

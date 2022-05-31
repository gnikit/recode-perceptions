import torch

from train.model_builder import MyCNN

# def test_preprocessing_pp():
#     """Test split size
#     """
#     df_train, df_val, df_test = datautils.pp_process_input(
#         "50a68a51fdc9f05596000002",
#         os.path.dirname(os.getcwd()),
#         "input/images/",
#         oversample=True,
#         verbose=True,
#     )


# def test_dataset_generator():
#     """Test pre-processing in [0,1]
#     and correct image labels"""


def test_model_builder():
    """Test random input
    passes through network
    and output shape is 2"""
    model = MyCNN()
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    # =================================
    # TEST SUITE
    # =================================
    # Check the length of the returned object
    assert len(out) == 2


# def test_one_training_epoch():
#     """Tests one forward and on backward pass
#     """

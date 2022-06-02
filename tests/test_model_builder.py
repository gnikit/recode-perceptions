import torch


def test_model_builder():
    from deep_cnn.model_builder import MyCNN

    """Test random input
    passes through network
    returning a single prediction"""
    model = MyCNN()
    x = torch.randn(1, 3, 224, 224)
    out = model(x)
    # =================================
    # TEST SUITE
    # =================================
    # Check the length of the returned object
    assert len(out) == 1

from unittest.mock import patch

from deep_cnn.utils import argument_parser


def test_train_model(root_dir, test_data, metadata):
    from deep_cnn.train_model import main

    opt = argument_parser(
        [
            "--root_dir=" + str(root_dir) + "/",
            "--data_dir=" + str(test_data) + "/",
            "--metadata=" + str(metadata),
            "--epochs=1",
            "--batch_size=1",
            "--run_name=test",
        ]
    )

    with patch("torch.save") as mock_save:
        main(opt)

    mock_save.assert_called()

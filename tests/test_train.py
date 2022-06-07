from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def test_training_epoch(root_dir, test_data, study, params, metadata):
    """tests dataset iterator"""
    import deep_cnn.train as train
    from deep_cnn.dataset_generator import dataloader
    from deep_cnn.datautils import pp_process_input
    from deep_cnn.model_builder import MyCNN

    df_train, df_val, _ = pp_process_input(
        root_dir=root_dir,
        data_dir=test_data,
        metadata=metadata,
        oversample=False,
        verbose=False,
        perception_study=study,
    )
    train_dataloader = dataloader(
        df_train, Path(root_dir, test_data), "resnet", "train", params
    )
    val_dataloader = dataloader(
        df_val, Path(root_dir, test_data), "resnet", "val", params
    )
    model = MyCNN()

    optimizer = torch.optim.Adam(model.parameters(), 0.001)
    loss_fn = nn.MSELoss()

    train_val_loss = train.train(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        scheduler=None,
        loss_fn=loss_fn,
        epochs=1,
        device="cpu",
        save_model=Path(root_dir, "outputs/models/test.pt"),
        wandb=False,
    )
    # =================================
    # TEST SUITE
    # =================================
    # Check train loss is not none and val loss is np.nan
    assert train_val_loss["train_loss"][0] != np.nan
    assert train_val_loss["val_loss"][0] is np.nan

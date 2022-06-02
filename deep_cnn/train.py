import torch
from tqdm import tqdm

"""
Contains functions for training and testing a PyTorch model.
"""


def train_step(
    epoch: int,
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.Optimizer,
    device: torch.device,
):
    """Trains the PyTorch model end-to-end for one epoch. Steps are:
    Forward pass, backward pass, loss calculation, optimizer step

    Args:
    model: a PyTorch model for training.
    dataloader: A DataLoader instance for the model to be trained on.
    loss_fn: A PyTorch loss function to minimise.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    device: A target device to compute on (e.g. "cuda" or "cpu").
    """

    # Put model in train mode
    model.train()

    # Setup train loss over epoch
    running_loss = 0

    # loop over training batches using timer tqdm
    with tqdm(train_dataloader, unit="batch") as tepoch:
        for data, target in tepoch:
            tepoch.set_description(f"Epoch {epoch}")

            # Format expected input dimensions and send data to device
            train_x = data.to(device)
            y = target.unsqueeze(dim=1).to(device)

            # 1. Forward Pass
            output = model.forward(train_x)

            # 2. Calculate and accumulate loss
            loss = loss_fn(output.float(), y.float())
            running_loss += loss.detach().item()

            # 3. Optimzer zero grad
            optimizer.zero_grad(set_to_none=False)

            # 4. Loss backprop
            loss.backward()

            # 5. Optimizer step
            optimizer.step()
            tepoch.set_postfix(loss=loss.detach().item())

    # 6. Optimizer Step
    scheduler.step()

    # Adjust metrics to get average loss per batch
    avg_train_loss = running_loss / (len(train_dataloader))
    return avg_train_loss


def test_step(
    model: torch.nn.Module,
    test_dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
):
    """Tests the PyTorch model end-to-end for one epoch. Steps are:
    Forward pass and loss calculation.

    Args:
    model: a PyTorch model for testing.
    dataloader: A DataLoader instance for the model to be trained on.
    loss_fn: A PyTorch loss function to calculate loss on test data.
    device: A target device to compute on (e.g. "cuda" or "cpu").
    """

    # Put model in eval
    model.eval()

    # Setup train loss over epoch
    running_loss = 0

    # loop over val/test batches
    for i, data in enumerate(test_dataloader):
        # Format expected input dimensions and send data to device
        test_x = data[0].to(device)

        y = data[1].unsqueeze(dim=1).to(device)

        # 1. Forward Pass
        output = model.forward(test_x)

        # 2. Calculate and accumulate loss
        loss = loss_fn(output.float(), y.float())
        running_loss += loss.detach().item()

    # Adjust metrics to get average loss and accuracy per batch
    avg_test_loss = running_loss / (len(test_dataloader))
    return avg_test_loss


def train(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    epochs: int,
    device: torch.device,
    save_model: str,
    wandb: bool,
):
    """Trains PyTorch model and reports validation accuracy

    Passes a target PyTorch models through train and validation set
    functions for a number of epochs, training and validating the model
    in the same epoch loop.
    Calculates, prints and stores evaluation metrics throughout.

    Args:
    model: A PyTorch model to be trained and validated.
    train_dataloader: A DataLoader instance for the model to be trained on.
    val_dataloader: A DataLoader instance for the model to be validated on.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    scheduler: A PyTorch scheduler to decrease learning rate over epochs.
    loss_fn: A PyTorch loss function to calculate loss on both datasets.
    epochs: An integer indicating how many epochs to train for.
    device: A target device to compute on (e.g. "cuda" or "cpu").
    Returns:
    A dictionary of training and testing loss as well as training and
    validation accuracy metrics. Each metric has a value in a list for
    each epoch.
    In the form: {train_loss: [...],
              val_loss: [...],
    For example if training for epochs=2:
             {train_loss: [2.0616, 1.0537],
              val_loss: [1.2641, 1.5706],
    """
    # Create empty results dictionary
    results = {
        "train_loss": [float],
        "val_loss": [float],
    }

    # Loop through training and testing steps for a number of epochs
    for epoch in range(epochs):
        train_loss = train_step(
            epoch=epoch,
            model=model,
            train_dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )
        print("Calculating validation loss")
        val_loss = test_step(
            model=model, test_dataloader=val_dataloader, loss_fn=loss_fn, device=device
        )

        # Print out what's happening
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"val_loss: {val_loss:.4f} | "
        )

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["val_loss"].append(val_loss)

        if wandb is True:
            pass
            # wandb.log(
            #     {
            #         "loss_train": train_loss,
            #         "loss_val": val_loss,
            #     }
            # )

    if save_model is not None:
        state = {
            "epoch": epochs,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        torch.save(state, save_model + ".pt")

    # Return the filled results at the end of the epochs
    return results

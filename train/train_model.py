import argparse
import os
from timeit import default_timer as timer

import datautils as datautils
import torch
import torch.nn as nn
from dataset_generator import CustomImageDataset
from model_builder import MyCNN
from torch.utils.data import DataLoader

import train as train

# import wandb

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", default=20, type=int, help="number of training epochs")
parser.add_argument("--batch_size", default=128, type=int, help="batch size for SGD")
parser.add_argument("--model", default="resnet101", type=str, help="model_loaded")
parser.add_argument(
    "--pre", default="resnet", type=str, help="pre processing for image input"
)
parser.add_argument(
    "--oversample", default=True, type=bool, help="whether to oversample"
)
parser.add_argument(
    "--root_dir",
    default=os.getcwd(),
    help="path to recode-perceptions",
)
parser.add_argument("--lr", default=1e-3, type=float, help="learning rate")
parser.add_argument(
    "--study_id",
    default="50a68a51fdc9f05596000002",
    type=str,
    help="perceptions_1_to_6",
)
parser.add_argument(
    "--run_name",
    default="default",
    type=str,
    help="unique name to identify hyperparameter choices",
)
parser.add_argument("--data_dir", default="input/images/", type=str, help="dataset")


def main():
    # parse arguments
    global args
    opt = parser.parse_args()

    # detect devices
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on %s device" % device)

    # # WANDB for HO
    # id = '%s' % opt.wandb_name
    # wandb.login(key='')
    # wandb.init(id = id, project='place_pulse_phd', entity='emilymuller1991')

    # load image metadata
    df_train, df_val, df_test = datautils.pp_process_input(
        opt.study_id,
        opt.root_dir,
        opt.data_dir,
        oversample=opt.oversample,
        verbose=True,
    )

    # create dataloaders
    training_gen = CustomImageDataset(
        df_train, root_dir=opt.root_dir + opt.data_dir, transform=opt.pre
    )
    validation_gen = CustomImageDataset(
        df_val, root_dir=opt.root_dir + opt.data_dir, transform=opt.pre
    )
    test_gen = CustomImageDataset(
        df_test, root_dir=opt.root_dir + opt.data_dir, transform=opt.pre
    )
    params = {
        "batch_size": opt.batch_size,
        "shuffle": True,
        "num_workers": 1,
        "pin_memory": True,
        "drop_last": True,
    }
    train_dataloader = DataLoader(training_gen, **params)
    validation_dataloader = DataLoader(validation_gen, **params)
    test_dataloader = DataLoader(test_gen, **params)

    print(
        "There are %s images in the training set"
        % str(train_dataloader.__len__() * opt.batch_size)
    )
    print(
        "There are %s images in the validation set"
        % str(validation_dataloader.__len__() * opt.batch_size)
    )
    print(
        "There are %s images in the test set"
        % str(test_dataloader.__len__() * opt.batch_size)
    )

    # initialise model
    model = MyCNN()
    model.to(device)
    print("Model loaded with %s parameters" % str(model.count_params()))

    # Set up Loss and Optimizer
    optimizer = torch.optim.Adam(model.parameters(), opt.lr)

    def lambda_decay(epoch):
        # defines learning rate decay
        return opt.lr * 1 / (1.0 + (opt.lr / opt.epochs) * epoch)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda_decay)
    loss_fn = nn.MSELoss()

    # Start the timer
    start_time = timer()

    # Train model
    train_val_loss = train.train(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=validation_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        epochs=opt.epochs,
        device=device,
        save_model=opt.root_dir + "outputs/models/" + opt.run_name,
        wandb=False,
    )

    # End the timer and print out how long it took
    end_time = timer()
    print(f"Model trained in: {end_time-start_time:.3f} seconds")

    # Get Test Performance
    test_loss = train.test_step(
        model=model,
        test_dataloader=validation_dataloader,
        loss_fn=loss_fn,
        device=device,
    )
    print(f"Model tested in: {timer()-end_time:.3f} seconds")

    print(
        "LOSS train {} valid {} test {}".format(
            train_val_loss["train_loss"][-1], train_val_loss["val_loss"][-1], test_loss
        )
    )


if __name__ == "__main__":
    main()

# PLOTS
# plot training loss
# plot validation loss
# plot test prediction histogram


# y[i] = np.squeeze(toutputs.cpu().detach().numpy())
# y_true[i] = np.squeeze(tlabels.numpy())
# y = y[y != 0]
# y_true = y_true[y_true != 0]
# avg_testloss = running_tloss/(i+1)
# prediction_hist(y.flatten(), y_true.flatten(), opt.model + '
# _epochs_' + str(opt.epochs) + '_lr_' + str(opt.lr)  + str(opt.oversample)
#  + str(opt.study_id), opt.prefix )
# print('LOSS train {} valid {} test {}'.format(avg_tloss, avg_vloss, avg_testloss))

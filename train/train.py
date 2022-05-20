import time
import numpy as np 
import pandas as pd
import torch
import torch.nn as nn
import argparse
import wandb
# network
from resnet101 import MyResNet 
# data
import dataset 
# dataloader
from my_classes_pytorch import CustomImageDataset 
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from plots import prediction_hist
import gc
# import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', default=20, type=int, help='number of training epochs')
parser.add_argument('--batch_size', default=128, type=int, help='batch size for SGD') 
parser.add_argument('--model', default='resnet101', type=str, help='model_loaded')
parser.add_argument('--pre', default='keras', type=str, help='pre_processing')
parser.add_argument('--oversample', default=True, type=bool, help='whether to oversample')
parser.add_argument('--prefix', default='/rds/general/user/emuller/home/emily/phd/', help='when using RDS')
parser.add_argument('--lr', default=1e-3,type=float, help='number of latent dimensions')
parser.add_argument('--study_id', default='50a68a51fdc9f05596000002', type = str, help='perceptions_1_to_6')
parser.add_argument('--wandb_name', default='dummy',type=str, help='number of latent dimensions')
parser.add_argument('--data', default= '006_place_pulse/place-pulse-2.0/image/images/', type=str, help='dataset')
opt = parser.parse_args()
print(opt)

# Detect CUDA devices
use_cuda = torch.cuda.is_available()                   # check if GPU exists
device = torch.device("cuda" if use_cuda else "cpu")

# WANDB for HO
id = '%s' % opt.wandb_name
wandb.login(key='')
wandb.init(id = id, project='place_pulse_phd', entity='emilymuller1991')

# INITIALISE MODEL
model = MyResNet()
model.to(device)
# here is the structure
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print ('Model loaded with %s parameters' % str(pytorch_total_params) )

# PREPARING DATA
df_img = dataset.format_img_ids(opt.prefix, opt.data, opt.path)
df_img = dataset.add_qscore(opt.prefix, df_img, opt.study_id)
df_img = dataset.scale_data(1,10, df_img)
# split train, test
df_train = df_img.sample(frac=0.7)
df_val = df_train.sample(frac=0.07)
df_train = df_train.drop(df_val.index)
df_test = df_img.drop(df_train.index)
# oversample
df_train = dataset.oversample(df_train)
# plot histograms
df_train['trueskill.score_norm'].hist()
df_val['trueskill.score_norm'].hist()
df_test['trueskill.score_norm'].hist()

# DATA LOADER
preprocess = transforms.Compose([
        transforms.Lambda(lambda image: torch.from_numpy(np.array(image).astype(np.float64)/255)),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

params = {'batch_size': opt.batch_size,
        'shuffle': True,
        'num_workers': 2,
        'pin_memory': True,
        'drop_last': True}

start_data_loading = time.time()

training_gen = CustomImageDataset(df_train, prefix=opt.prefix+opt.data, transform=preprocess, target_transform=None)
validation_gen = CustomImageDataset(df_val, prefix=opt.prefix+opt.data, transform=preprocess, target_transform=None)
test_gen = CustomImageDataset(df_test, prefix=opt.prefix+opt.data, transform=preprocess, target_transform=None)
train_dataset = DataLoader(training_gen, **params) 
validation_dataset = DataLoader(validation_gen, **params) 
test_dataset = DataLoader(test_gen, **params)
print ('Finished loading data in %s seconds' % str(time.time() - start_data_loading))

print ('There are %s images in the training set' % str(training_gen.__len__()) )
print ('There are %s images in the validation set' % str(validation_gen.__len__()) )
print ('There are %s images in the test set' % str(test_gen.__len__()) )

# INITIALISE BACKPROP
model_params = list(model.parameters())
optimizer = torch.optim.Adam(model_params, opt.lr)
lambda_decay = lambda epoch: opt.lr * 1 / (1. + (opt.lr/opt.epochs) * epoch)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda_decay)

# TRAINING
criterion = nn.MSELoss()
training_outcomes = {}
init_time = time.time()

for epoch in range(1,opt.epochs+1):
    start_time = time.time()
    epoch_time = time.time()
    running_loss = 0
    model.train(True)
    for i, data in enumerate(train_dataset):
        train_x = data[0]
        y = data[1].unsqueeze(dim=1)
        
        optimizer.zero_grad(set_to_none=False)
        output = model.forward(train_x.to(device))
        loss = criterion(output.float() ,y.float().to(device))
        loss.backward()

        optimizer.step()
    
        # print statistics
        running_loss += loss.detach().item()
    
    scheduler.step()
    loss = loss.detach().item()
    avg_tloss = running_loss/ (i + 1)
    print ('EPOCH::training completed in %s for epoch %s' % (time.time() - start_time, epoch))

    # validation loss
    model.train(False)
    running_vloss = 0
    for i, vdata in enumerate(validation_dataset):
        vinputs = vdata[0]
        vlabels = vdata[1].unsqueeze(dim=1)
        voutputs = model.forward(vinputs.to(device))
        vloss = criterion(voutputs.float() , vlabels.float().to(device)).detach().item()
        running_vloss += vloss

    avg_vloss = running_vloss / (i + 1)
    print ('EPOCH::validation completed in %s for epoch %s' % (time.time() - start_time, epoch))

    print('LOSS train {} valid {}'.format(avg_tloss, avg_vloss))

    # save training and validation loss
    print ('Epoch finished in %s seconds' % str(time.time() - epoch_time))

    training_outcomes[epoch] = [int(epoch), avg_tloss, avg_vloss, epoch_time]
    df = pd.DataFrame(training_outcomes).T
    df.columns = ['epoch', 'loss_train', 'loss_val', 'time']
    df.to_csv(opt.prefix + '006_place_pulse/place-pulse-2.0/outputs/epoch_loss_history_%s.csv' % (opt.model + '_epochs_' + str(opt.epochs) + '_lr_' + str(opt.lr) + str(opt.oversample) + str(opt.study_id)))

    wandb.log( {
        'loss_train': avg_tloss,
        'loss_val': avg_vloss,
    })

    # save model checkpoint
    filepath = opt.prefix + '006_place_pulse/place-pulse-2.0/model/torch_resnet/model_checkpoint_epoch_%s_%s.pt' % ( str(opt.epochs), opt.wandb_name )
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(state, filepath)

    torch.cuda.empty_cache()
    gc.collect() 

end_time = init_time = time.time()
print ('Model trained in %s hours' % str((init_time - start_time)/3600))

########################################################## CLEAR GPU MEMORY

# TESTING LOSS
model.train(False)
running_tloss = 0
y = np.zeros((len(test_gen), opt.batch_size))
y_true = np.zeros((len(test_gen), opt.batch_size ))
for i, tdata in enumerate(test_dataset):
    test_x = tdata[0]
    tlabels = tdata[1].unsqueeze(dim=1)
    toutputs = model.forward(test_x.to(device))
    tloss = criterion(toutputs.float(), tlabels.float().to(device)).detach().item()
    running_tloss += tloss
    y[i] = np.squeeze(toutputs.cpu().detach().numpy())
    y_true[i] = np.squeeze(tlabels.numpy())
y = y[y != 0]
y_true = y_true[y_true != 0]
avg_testloss = running_tloss/(i+1)
prediction_hist(y.flatten(), y_true.flatten(), opt.model + '_epochs_' + str(opt.epochs) + '_lr_' + str(opt.lr)  + str(opt.oversample) + str(opt.study_id), opt.prefix )
print('LOSS train {} valid {} test {}'.format(avg_tloss, avg_vloss, avg_testloss))
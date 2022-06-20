# Training a Convolutional Neural Network in PyTorch

In this exercise we will fine-tune a pretrained CNN to learn to predict perception scores. Before we dive into the components of the model, we will just get the code up and running with the default settings.

## Contents

- Training Locally
- Export to HPC
- Hyperparameter Optimisation

### Training Locally

Locally means using the current hardware in your computer or laptop to run model training. If you have not yet done so, now is a good time to clone this repository into your local drive (see Getting Started).

Let's check the model training runs locally (albeit slowly without a GPU). From root_dir run:

```sh
root_dir$ python deep_cnn/train_model.py\
--epochs=1                              \
--batch_size=56                         \
--model='resnet101'                     \
--lr=1e-3                               \
--data_dir=data_dir                     \
--root_dir=root_dir                     \
```

where root_dir is your local path to recode-perceptions, and data_dir is your path to input/places365standard_easyformat/places365_standard. If you have not yet downloaded the images, you can run the test images, tests/test_input/test_images/. Similarly for metadata.

You should see the following output in your terminal:

```
Running on cuda device
Epoch 0:   0%|▍                                                                                 | 20/4371 [00:12<42:52,  1.69batch/s, loss=8.17]
```

If you have a gpu locally, you will also see 'Running on cuda device'. However, this will be replaced by cpu if no gpu device is found. The model is training through its first epoch batch by batch. This one epoch is expected to take 42 minutes to complete.

Once finished, the full log can be found in outputs/logger/default.log.

### Export to HPC



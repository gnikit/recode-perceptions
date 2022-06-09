# Training a Convolutional Neural Network in PyTorch

In this exercise we will fine-tune a pretrained CNN to learn to predict perception scores. Before we dive into the components of the model, we will just get the code up and running with the default settings.

## Contents

- Training Locally
- Export to HPC
- Design choices
- Hyperparameter Optimisation

### Training Locally

Locally means using the current hardware in your computer or laptop to run model training. If you have not yet done so, now is a good time to clone this repository into your local drive (see Getting Started).

Let's check the model training runs locally (albeit slowly without a GPU). From root_dir run:

```sh
root_dir$ python deep_cnn/train_model.py\
--epochs=1                              \
--batch_size=1                          \
--model='resnet101'                     \
--oversample=True                       \
--lr=1e-3                               \
--study_id='50a68a51fdc9f05596000002'   \
--data_dir=data_dir                     \
--root_dir=root_dir                     \
--metadata=meta_dir                     \
```

where root_dir is your local path to recode-perceptions, and data_dir is your path to recode-perceptions/input/images/. If you have not yet downloaded the images, you can run the test images, tests/test_input/test_images/. Similarly for metadata.

You should see the following output ...

The model has been trained, but for only one epoch. What do you notice about the test accuracy after just one epoch? Let's now migrate the programme to the HPC so we can utilise the GPUs available for faster implementation.

### Export to HPC



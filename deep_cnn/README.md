# Training a Convolutional Neural Network in PyTorch

In this exercise we will fine-tune a pretrained CNN to learn to predict perception scores. Before we dive into the components of the model, we will just get the code up and running with the default settings.

## Contents

- Training Locally
- Hyperparameter Optimisation

### Training Locally

Locally means using the current hardware in your computer or laptop to run model training. If you have not yet done so, now is a good time to clone this repository into your local drive (see Getting Started).

Let's check the model training runs locally (albeit slowly without a GPU). From root_dir run:

```sh
root_dir$ python3 -m deep_cnn           \
--epochs=1                              \
--batch_size=56                         \
--model='resnet101'                     \
--lr=1e-3                               \
--data_dir=data_dir                     \
--root_dir=root_dir                     \
--wandb=False                           \
```

where root_dir is your local path to recode-perceptions, and data_dir is your path to input/places365standard_easyformat/places365_standard. If you have not yet downloaded the images, you can run the test images, tests/test_input/test_images/. Similarly for metadata.

You should see the following output in your terminal:

```
Running on cuda device
Epoch 0:   0%|▍                                                                                 | 20/4371 [00:12<42:52,  1.69batch/s, loss=8.17]
```

If you have a gpu locally, you will also see 'Running on cuda device'. However, this will be replaced by cpu if no gpu device is found. The model is training through its first epoch batch by batch. This one epoch is expected to take 42 minutes to complete.

Once finished, the full log can be found in outputs/logger/default.log.

### Hyperparameter Optimisation

Let's take a look at the design choices which can be made for model training.

| Parameter | Description | Trade Offs | References |
|-----------|-------------|------------|------------|
|Epochs     |Determine the number of times to run through the entire training batch. | Typically there is an inflection point where decreases in loss are marginal. Continued increase after this reflects overfitting | [Bias-Variance Trade-Off](https://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/lecturenote12.html)  |
|Batch size           |Number of samples for each training step| Smaller batch sizes increase performance of stochastic gradient descent algorithms while also preserving memory by loading in batches. Larger batch size can speed up computation. | [Batch-size](https://stats.stackexchange.com/questions/164876/what-is-the-trade-off-between-batch-size-and-number-of-iterations-to-train-a-neu)|
|Learning Rate|
|Model          |Pre-trained model| Can vary by size, depth, structure and trainable parameters | Smaller models are faster to train while deeper model typically achieve higher levels of abstraction. Models with dropout can avoid overfitting,         | [PyTorch pre-trained models](https://pytorch.org/vision/stable/models.html) |
|Pre-processing  | Image pre-processing required for model input | Pre-trained models have parameters trained within a given range and perform better when the target dataset distribution is closer matched to the source dataset distribution | |

### Export to HPC



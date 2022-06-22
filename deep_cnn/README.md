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
--model='resnet18'                      \
--lr=1e-3                               \
--data_dir=data_dir                     \
--root_dir=root_dir                     \
--wandb=False                           \
```

A brief description of all arguments is returned with the following:

```
root_dir $ python3 -m deep_cnn -h
usage: __main__.py [-h] [--epochs EPOCHS] [--batch_size BATCH_SIZE]
                   [--model MODEL] [--pre PRE] [--root_dir ROOT_DIR] [--lr LR]
                   [--run_name RUN_NAME] [--data_dir DATA_DIR] [--wandb WANDB]

optional arguments:
  -h, --help            show this help message and exit
  --epochs EPOCHS       number of training epochs
  --batch_size BATCH_SIZE
                        batch size for SGD
  --model MODEL         pre-trained model
  --pre PRE             pre-processing for image input
  --root_dir ROOT_DIR   path to recode-perceptions
  --lr LR               learning rate
  --run_name RUN_NAME   unique name to identify hyperparameter choices
  --data_dir DATA_DIR   path to input data
  --wandb WANDB         track progress in wandb.ai
```

where root_dir is your local path to recode-perceptions, and data_dir is your path to input/places365standard_easyformat/places365_standard. If you have not yet downloaded the full image dataset, but you want to run this script locally, you can run the test images, tests/test_input/test_images/.

You should see the following output in your terminal:

```
Running on cuda device
Epoch 0:   2%|██▎                                                                                                                | 35/1697 [00:19<15:28,  1.79batch/s, loss=3.06]
```

If you have a gpu locally, you will also see 'Running on cuda device'. However, this will be replaced by cpu if no gpu device is found. The model is training through its first epoch batch by batch. This one epoch is expected to take 15 minutes to complete.

Once finished, the full log can be found in the folder outputs/logger/. You will find an annotated version in the repository.

### Hyperparameter Optimisation

Let's take a look at the design choices which can be made for model training.

| Parameter | Description | Trade Offs | References |
|-----------|-------------|------------|------------|
|Epochs     |Determine the number of times to run through the entire training batch. | Typically there is an inflection point where decreases in loss are marginal. Continued increase after this reflects overfitting | [Bias-Variance Trade-Off](https://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/lecturenote12.html)  |
|Batch size           |Number of samples for each training step| Smaller batch sizes increase performance of stochastic gradient descent algorithms while also preserving memory by loading in batches. Larger batch size can speed up computation. | [Batch-size](https://stats.stackexchange.com/questions/164876/what-is-the-trade-off-between-batch-size-and-number-of-iterations-to-train-a-neu)|
|Learning Rate| Step size of parameter update at each iteration | A larger learning rate requires fewer epochs but can easily result in unstable results such as local minima. Smaller learning rates can fail to find an optima at all. Adaptive learning rates consider large step sizes in the earlier epochs of training and reduces the step size in later epochs | [Learning Rates](https://machinelearningmastery.com/understand-the-dynamics-of-learning-rate-on-deep-learning-neural-networks/) |
|Model          |Pre-trained model| Can vary by size, depth, structure and trainable parameters | Smaller models are faster to train while deeper model typically achieve higher levels of abstraction. Models with dropout can avoid overfitting,         | [PyTorch pre-trained models](https://pytorch.org/vision/stable/models.html) |
|Pre-processing  | Image pre-processing required for model input | Pre-trained models have parameters trained within a given range and perform better when the target dataset distribution is closer matched to the source dataset distribution | |

There is never a one-rule-fits-all-approach to training a deep neural network. Design choices can be guided by the domain task, the dataset size and distribution, hardware constraints, time constraints or all of the above. The wonder of moden software and computing is that the cycle of iterating on model choices has been sped up enormously:

![alt text](./learning/Images/image_tasks.png "Hyperparameter Optimisation Cycle")

There exists many different types of [search algorithms](https://en.wikipedia.org/wiki/Hyperparameter_optimization) for finding the optimal hyperparameters, such as gridded search, random search and bayesian optimisation. We would like to add apriori the wisdom of the crowd. There has been a lot written about deep learning hyperparameters so we don't need to go in blind. To help you configure some intervals, consider the following questions

1) How many epochs will ensure you have reached the global miniima?
2) How much memory constraints do you have on model and batch size?
3) What ratio of batch size to data samples/classes is considered a benchmark?
4) What is a typical learning rate for similar image classification tasks?
5) How can an adaptive learning rate be implemented?
6) Which model has shown best performance on benchmark datasets? Is there a PyTorch version to download pre-trained weights?
7) Does this model require certain pre-processing? If yes, you will have to add the preprocessing to dataset_generator.py

Once you have answered the questions above, you will have a constraint on what is reasonable to test.

When performing hyperparamter optimisation, we will not evaluate our test performance during training. This test set will be a hold-out set used only to evaluate the performance of our final model after training. Instead we evaluate the performance of the model during training on the validation set. This is done typically to avoid overfitting to the test set during hyperparamter optimisation. Our goal in training is to test the generalisability of our training paradigm and the test-set allows us to do this.

#### Hyperparamter optimisation using wandb.ai

We will use weights and biases to track our model training and validation. This platform uses a lightweight python package to log metrics during training. To use this, you will need to create an educational account at [https://wandb.ai/site](https://wandb.ai/site). Once you have configured your user credentials, create a new project called recode-perceptions. This folder will log all of our training runs. In order for python to get access to your personal user credentials, you will have to set them as environmental variables which python then accesses using os.getenv("password"). Set your environmental variables as follows:

```
root_dir$ export $WB_KEY=weights_and_biases_login_key
export $WB_project="recode-perceptions"
export $WB_USER="weights_and_biases_user"
```

If you now run the scripts with --wandb=True, you should begin to see the metrics being tracked on the platform:

![alt text](./learning/Images/wandb.png "Logging metrics using wandb")

### Export to HPC



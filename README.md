# ReCoDE Perceptions - An end-to-end supervised learning task

Recode Perceptions is a pyTorch implementation of a deep convolutional neural network model trained on place pulse perceptions data, developed by Emily Muller.

This model is trained on 70K images with associated perception scores to be able to predict safety, beauty and wealth across new unseen images of the urban environment. Through transfer learning the model can achieve prediction accuracies within plus or minus one decile.

## Overview

- demo/ contains a notebook showing how to import a dataset, load a model and run it on that dataset making forward and backward passes.
- train/ contains code for full model training pipeline.
- model/ contains an already trained model.
- input/ contains the original image data (to download - see below) and metadata.
- output/ folder for model training outputs.
- tests/ contains unit tests for the codebase.

### Installation

We assume that you're using Python 3.7 with pip installed and commands are unix.

#### Setting up a virtual environment

We will set up a virtual environment for running our scripts. In this case, installing specific package versions will not interfere with other programmes we run locally as the environment is contained. Initially, let's set up a virtual environment with python version 3.7:

```
virtualenv --python=python3.7 venv
```

This will create a new folder named venv in your repository. We can activate this environment by running

```
source venv/bin/activate
```

Let's configure our environment by installing all the necessary packages located in requirements.txt

```
pip install -r requirements.txt
```

### Dataset

Then, run the download script to downloads all of the Place Pulse images (~3GB) from here and put them in the input/images/ directory:

```
wget -O /root/recode/input/images.zip https://www.dropbox.com/s/grzoiwsaeqrmc1l/place-pulse-2.0.zip?dl=1
```

where root is the path to the recode repository. Unzip/extract all files in the same location

```
unzip images.zip
```




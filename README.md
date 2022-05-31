# PyTorch implementation for end-to-end training of a deep learning model

Recode Perceptions is a pyTorch implementation of a deep convolutional neural network model trained on place pulse perceptions data, developed by Emily Muller.

This model is trained on 70K images with associated perception scores to be able to predict safety, beauty and wealth across new unseen images of the urban environment. Through transfer learning the model can achieve prediction accuracies within plus or minus one decile.

## Overview
- learning/ contains learning material to help you understand the science of deep learning and the substantive background of urban perception.
- demo/ contains a notebook showing how to import a dataset, load a model and run it on that dataset making forward and backward passes.
- train/ contains code for full model training pipeline.
- model/ contains an already trained model.
- input/ contains the original image data (to download - see below) and metadata.
- output/ folder for model training outputs.
- tests/ contains unit tests for the codebase.

## Getting started
Clone this repository into your local drive. root_dir will now refer to the /local_path/to_this/directory/recode-perceptions.

### Installation

We assume that you're using Python 3.7 with pip installed.

### Setting up a virtual environment

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

The dataset can be downloaded from dropbox. Run wget (below) to downloads all of the Place Pulse images (~3GB) and put them in the input/images/ directory:

```
wget -O /root/recode/input/images.zip https://www.dropbox.com/s/grzoiwsaeqrmc1l/place-pulse-2.0.zip?dl=1
```

where root is the path to the recode repository. Unzip/extract all files in the same location

```
unzip images.zip
```
In addition, unzip input/meta.zip to extract the label information.



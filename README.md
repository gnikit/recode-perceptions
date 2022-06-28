# PyTorch implementation for end-to-end training of a deep learning model

Recode Perceptions is a pyTorch implementation of a deep convolutional neural network model trained on Places365 data, developed by Emily Muller.

This model is trained on a subset of 100K images which have outcome labels that are associated to factors which are relevant for environmental health.

## How to Use this Repository

This repository has 3 core learning components:

| Title | Description | Location | Key Learning Objectives  |
|--------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Introduction to Environmental Health and Imagery | This is a short video, introducing the domain, methods and describing some of the pioneering work in this field                                                                                           | [Video](learning/)                              | Introduction to the field. Understand different methods. Understand different types of data. Be aware of seminal research.                                                                                                                                                                                                       |
| Foundations of Deep CNN's using PyTorch          | A Jupyter Notebook familiarising students with core components of deep learning framework using PyTorch.                                                                                                  | [Jupyter Notebook](learning/convolutional_neural_networks.ipynb) | Be aware of different types of Computer Vision tasks. Be able to explain what a convolutional layer does and how it's different from a fully-connected layer. Identify different components of a CNN. Load an image dataset in PyTorch. Load a pre-trained model in PyTorch. Be able to use PyTorch to train a model on a dataset |
| deep_cnn  | This module contains all the code needed to fine-tune a deep neural network on the Places365 classification task. Detailed documentation is provided in the folder README.md but requires set up (below). | [deep_cnn](deep_cnn/README.md)                                    | Use terminal for executing python scripts Train a PyTorch model and visualise results. Export training to the HPC. Implement bach script Iterate on model hyperparameters to optimise model. |

The suggested way to use this repository is as follows:

- Continue with set-up as detailed below.
- Complete learning materials 1 ([Video](learning/)) and 2 ([Jupyter Notebook](learning/convolutional_neural_networks.ipynb)).
- Continue to model training in 3 ([deep_cnn](deep_cnn/README.md)).

## Overview

- learning/ contains learning material to help you understand the science of deep learning and the substantive background of urban perception.
- deep_cnn/ contains code for full model training pipeline.
- model/ contains an already trained model.
- input/ contains the original image data (to download - see below) and metadata.
- output/ folder for model training outputs.
- tests/ contains unit tests for the codebase.

```
recode-perceptions
│   README.md
│   .pre-commit-config.yaml             # configure pre-commit tools for formatting codebase
│   setup.cfg                           # set-up for pre-commit
│   requirements-dev.txt                # python packages for development
│   requirements.txt                    # python packages for running programme
│   environment.sh                      # HPC environment set-up (see)
│   submit.pbs                          # job submission for HPC (see)
│
└───deep_cnn                            # module for model training (see)
│   │   __init__.py
│   │   __main__.py
│   │   logger.py
│   │   utils.py
│   │   dataset_generator.py
│   │   model_builder.py
│   │   train.py
│   │   train_model.py
│
└───input                               # folder to download images
│   │   keep.txt                        # images to remove
│   └───places365standard_easyformat    # images downloaded (see)
│   │   │   ...                         # metadata
│   │   └───places365_standard
│   │   │   └───train
│   │   │   └───val
│
└───learning
│   │   convolutional_neural_networks.ipynb
│
└───outputs
│   └───logger                          # logging output from model training
│   └───models                          # save model checkpoints
│   └───results                         # save model training metrics
│
└───tests                               # folder for testing
│   └───places_test_input
│   │   ...
```

## Getting started

Clone this repository into your local drive.

```sh
git clone https://github.com/ImperialCollegeLondon/recode-perceptions.git
cd recode-perceptions
```

### Installation

We assume that you're using Anaconda + Python 3.7.

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

The final command requires you to have python installed and will install all the packages contained within the requirements.txt file. If an error occurs at this stage, don't panic! Often, this will be a result of python version conflict for a certain package/packages. Make sure that python3.7 version is being used.

### Dataset

The dataset can be downloaded from dropbox. Run wget (below) to download all of the Places365 train/val images (~21GB) and put them in the input/ directory:

```
wget -O /root_dir/recode/input/places365standard_easyformat.tar http://data.csail.mit.edu/places/places365/places365standard_easyformat.tar
```

where root_dir is the path to the recode repository. Unzip/extract all files in the same location

```
tar -xvf places365standard_easyformat.tar
```

We will remove categories which are not substantively interesting for the theme of environmental health. To do so run

```
cd input
GLOBIGNORE=$(paste -s -d : keep.txt)
cd places365standard_easyformat/places365_standard/train
rm -rf *
cd ../val
rm -rf *
unset GLOBIGNORE
```

GLOBIGNORE specifies folders which should be ignored when performing recursive deletes.

The datasets were not released by us and we do not claim any rights on them. Use the datasets at your responsibility and make sure you fulfill the licenses that they were released with. If you use any of the datasets please consider citing the original authors of [Places365](http://places2.csail.mit.edu/PAMI_places.pdf).

### Testing

To run all tests, install pytest. After installing, run

```
pytest tests/ -v
```

### License

This code and the pretrained model is licensed under the MIT license.



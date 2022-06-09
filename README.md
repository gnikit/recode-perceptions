# PyTorch implementation for end-to-end training of a deep learning model

Recode Perceptions is a pyTorch implementation of a deep convolutional neural network model trained on place pulse perceptions data, developed by Emily Muller.

This model is trained on 70K images with associated perception scores to be able to predict safety, beauty and wealth across new unseen images of the urban environment. Through transfer learning the model can achieve prediction accuracies within plus or minus one decile.

## Overview

- learning/ contains learning material to help you understand the science of deep learning and the substantive background of urban perception.
- demo/ contains a notebook showing how to import a dataset, load a model and run it on that dataset making forward and backward passes.
- deep_cnn/ contains code for full model training pipeline.
- model/ contains an already trained model.
- input/ contains the original image data (to download - see below) and metadata.
- output/ folder for model training outputs.
- tests/ contains unit tests for the codebase.

## Getting started

Clone this repository into your local drive. root_dir will now refer to the /local_path/to_this/directory/recode-perceptions.

```sh
git clone
cd recode-perceptions
pip install -r requirements.txt
```

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

The final command requires you to have python installed and will install all the packages contained within the requirements.txt file. If an error occurs at this stage, don't panic! Often, this will be a result of python version conflict for a certain package/packages. Make sure that python3.7 version is being used.

### Dataset

The dataset can be downloaded from dropbox. Run wget (below) to download all of the Place Pulse images (~3GB) and put them in the input/images/ directory:

```
wget -O /root_dir/recode/input/images.zip https://www.dropbox.com/s/grzoiwsaeqrmc1l/place-pulse-2.0.zip?dl=1
```

where root_dir is the path to the recode repository. Unzip/extract all files in the same location

```
unzip images.zip
```

In addition, unzip input/meta.zip to extract the label information. The datasets were not released by us and we do not claim any rights on them. Use the datasets at your responsibility and make sure you fulfill the licenses that they were released with. If you use any of the datasets please consider citing the original authors of [Place Pulse](https://arxiv.org/pdf/1608.01769.pdf).

### Testing

To run all tests, install pytest. After installing, run

```
pytest tests/ -v
```

### License

This code and the pretrained model is licensed under the MIT license.



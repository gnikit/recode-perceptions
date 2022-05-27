# Training a Convolutional Neural Network in PyTorch

In this exercise we will fine-tune a pretrained CNN to learn to predict perception scores. Before we dive into the components of the model, we will just get the code up and running with the default settings.

## Contents
- Training Locally 
- Export to HPC
- Design choices
- Hyperparameter Optimisation

<!-- Since we will now be working with the full model and full dataset, we will want to optimise the performance using a GPU.  -->
### Training Locally
Locally means using the current hardware in your computer or laptop to run model training. If you have not yet done so, now is a good time to clone this repository into your local drive.
```sh
git clone 
cd 
pip install -r requirements.txt
```
The final command requires you to have python installed and will install all the packages contained within the requirements.txt file. If an error occurs at this stage, don't panic! Often, this will be a result of python version conflict for a certain package/packages. Just head back to the main project README.md for instructions onn how to create a virtual environment with the correct python version.

Let's check the model training runs locally (albeit slowly):
```sh
python train_model.py                   \
--epochs=1                              \
--batch_size=50                         \
--model='resnet101'                     \
--oversample=True                       \
--lr=1e-3                               \
--study_id='50a68a51fdc9f05596000002'   \
--data_dir='input/test_images/'              \
--root_dir='/tf/misc/recode/'

```
You should see the following output ... The model has been trained, but for only one epoch. What do you notice about the test accuracy after just one epoch? Let's now migrate the programme to the HPC so we can utilise the GPUs available for faster implementation.

### Export to HPC



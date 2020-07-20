## Self Supervised Learning for Image Classification

Implementation of a paper that uses geometric transformations to extract features of an image without requiring these images to be labeled. 

Paper: https://arxiv.org/pdf/1803.07728.pdf

### Setting up the environment
`pip3 install vitrualenv` (if not already installed)
`virtualenv venv`
`source venv/bin/activate`
`pip3 install -r requirements.txt`

To deactivate the environment you are in run:
`source deactivate`

### Code Structure
Most of the code is written in the data.py, resnet.py, and rotnet.py.

`Rotnet.py` contains the training loop and the basic graph for the model. 
`Resnet.py` contains the implementation of the resnet model
`Data.py` contains all the data loading functions.

You can start training by running `main.py` with the following command:
`python3 main.py --config config.yaml --train --data_dir ./data/cifar-10-batches-py/ --model_number 1`

`config.yaml` contains the configuration file with all the hyperparameters. 

### Additional Details
#### Downloading the CIFAR-10 dataset
You can read more about the CIFAR-10 dataset here: https://www.kaggle.com/c/cifar-10
1. Go to this link https://www.cs.toronto.edu/~kriz/cifar.html
2. Right click on "CIFAR-10 python version" and click "Copy Link Address"
3. Go to your CLI and go into the `data` directory.
4. Run this cURL command to start downloading the dataset: `curl -O <URL of the link that you copied>`
5. To extract the data from the .tar file run: `tar -xzvf <name of file>` (type `man tar` in your CLI to see the different options for running the tar command).
**NOTE**: Each file in the directory contains a batch of images in CIFAR-10 that have been serialized using python's pickle module. You will have to first unpickle the data before loading it into your model.

#### Resnet18 Architecture
https://www.google.com/search?q=resnet+architecture&tbm=isch&source=iu&ictx=1&fir=nrwHYuY3M7ZNXM%253A%252CmlG8I6OjyTBN4M%252C_&vet=1&usg=AI4_-kRZVFcZ9REeELvn4BDXDpOJhFpNQg&sa=X&ved=2ahUKEwjd5NiphYjkAhVPKa0KHROtD3QQ9QEwBHoECAYQCQ#imgrc=eLRQQc-BgrBkxM:&vet=1



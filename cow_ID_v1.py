# Transfer learning demo for individual cow identification classification problem
# 4/26/2019
# Jake Decoto and Maggie Engler
# Initial version based on pytorch tutorial available here
# https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import helper_functions as hlp
from itertools import islice

# Constants
NUMCLASSES = 12 #Number of Classification Classes
NUM_EPOCHS = 25 #number of training epochs (baseline = 25)
MODEL_PATH = 'model.pt' #save path for model file
CONTINUE_FLAG = 0 #1=load last saved model file and continue training else start fresh from pre-trained pytorch model
DATA_DIR = 'data'
FULL_TRAIN_SIZE = 745 #Photos only

# Set matplotlib interactive mode
plt.ion()  

# NOTE: Per Pytorch documentation, pretrained Pytorch models (resnet18, alexnet, squeezenet, 
# vgg15, densenet, inception, googlenet) assume images of size (3,H,W) where H and W are at 
# least 224.  Images should be loaded in ragne of [0,1] and normalized using mean = [0.485, 
# 0.456, 0.406] and std = [0.229, 0.224, 0.225]

# Data augmentation and normalization for training
# Just normalization for validation

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Load Data and Apply Augmentation/Normalization
image_datasets = {x: datasets.ImageFolder(os.path.join(DATA_DIR, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=4)
              for x in ['val']}

# Setup device for processing, default is cpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Show examples of training data
# inputs, classes = next(iter(dataloaders['train'])) # Get a batch of training data
# out = torchvision.utils.make_grid(inputs) # Make a grid from batch
# hlp.imshow(out, title=[class_names[x] for x in classes])
# input('press <ENTER> to continue')

# Function for training model
def train_model(model, criterion, optimizer, scheduler, num_samples, num_epochs=25):
    since = time.time()
    subset_indices = np.random.choice(range(len(image_datasets['train'])), num_samples, replace=False)
    train_fraction = torch.utils.data.random_split(image_datasets['train'], [num_samples, len(image_datasets['train'])-num_samples])[0]
    dataloaders['train'] = torch.utils.data.DataLoader(train_fraction, batch_size=4,
    shuffle=True, num_workers=4)
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    print(dataset_sizes)
    class_names = image_datasets['train'].classes
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model, best_acc

# Function for displaying a selection of results
def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                print(i, outputs[j])
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('Pred: {}'.format(class_names[preds[j]] + ', Act: {}'.format(labels[j])))
                hlp.imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

# Initialize model
model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, NUMCLASSES)

# Continue from last saved model
if CONTINUE_FLAG == 1: 
    print('Loading Saved Model: ', MODEL_PATH)
    model_ft.load_state_dict(torch.load(MODEL_PATH))

# Assign model to device (CPU if local)
model_ft = model_ft.to(device)

# Set up optimize (note: all parameters are being optimized) 
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Setup learning rate decay schedule (note: LR by a factor of 0.1 every 7 epochs)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

# Train and return model
criterion = nn.CrossEntropyLoss()


model_ft, best_acc = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, FULL_TRAIN_SIZE, num_epochs=NUM_EPOCHS)

# Save model
torch.save(model_ft.state_dict(), MODEL_PATH)

# Visualize model
# visualize_model(model_ft)
# input('press <ENTER> to continue')
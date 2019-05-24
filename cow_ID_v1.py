# Individual Cattle Identification in Pictures Using Sparse Training Data
# 5/21/2019
# Jake Decoto and Maggie Engler

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
import shutil
from shutil import copyfile

# Constants
MODEL_PATH = 'models/model.pt' #save path for model file
CONTINUE_FLAG = 0 #1=load last saved model file and continue training else start fresh from pre-trained pytorch model
DATA_DIR = 'data'
EVAL_FLAG = 0 #If set to 1 perform eval on validation set only, do not train model (CONTINUE_FLAG must also be set to 1)

#Hyperparameters
NUM_EPOCHS = 50 #number of training epochs (baseline = 25)
CLASSES = ['004','005','006'] #['000','001','002','003','004','005','006','007','008','009','010','011'] #Which labelled classes to include
EPOCH_SPLIT = 190512 #YYMMDD epoch to split data between train (everything prior) and val/test (everything at or after epoch) 
VAL_TEST_SPLIT = 1 #Fraction of data in val/test that is assigned to val (1=All val, 0 = All test)

#Hyperpareters - Network
PRETRAIN_NET = 'ALEX' #Pretrained network to use for transfer learning (Choices: 'RES', 'ALEX', 'VGG')
FREEZE_PRIOR = True #Freeze parameters of pre-trained network (True or False), and only learn weights for added layer(s)
NUM_FREEZE = 1000 #Number of layers for (set to large number, like 1000 if desiring to freeze all pretrained layers)

#Hyperparameters - Optimizer
LEARNING_RATE = 1e-4 #learning rate
WEIGHT_DECAY = 0 #weight decay (L2 penalty)
OPTIMIZER = 'SGD' #Optimizer (Choices: 'ADAM', 'SGD')
SGD_SETTINGS = [True, 0.9] #For SGD optimizer only use nesterov (True or False) and momentum value (0-1)
LR_DECAY_FACTOR = 0.1 #Factor by which learning rate decay will take place.  Factor of LR_DECAY_FACTOR every LR_DECAY_EPOCHS  epochs.
LR_DECAY_EPOCHS = 7 

#Hyperpareters - Data Augmentation at Training Time
PROB_HORIZ = 0.5 #Probability of horizontal flip
PROB_GRAY = 0.5 #Probability of transormation to grayscale
COLOR_JITTER = [0.25, 0.25, 0.25, 0.25] #Random color jitter settings [brightness, contrast, saturation, hue] allowable rescaling ranges 0=Full, 1=NoChangeAllowed

#NOTE: It should be OK to not have a test set now, (i.e. VAL_TEST_SPLIT=1) since we can collect more data later to test our final model(s)
#against.  Also, for now I'd reccommend we use EPOCH_SPLI=190512 (May 12th) to ensure that all classes have data in both the train and val sets.

#Purge and rebuild temp data directory
try:
    shutil.rmtree('tmp')
except:
    pass
os.mkdir('tmp')
for x in ['train', 'val', 'test']:
    os.mkdir('tmp/' + x)
    for c in CLASSES:
        os.mkdir('tmp/' + x + '/' + c)

#Split data into train/val/test in the tmp directory
for root, dirs, file in os.walk(DATA_DIR):
    if '/' in root:
        c = root.split('/')[1] #class folder
        if c in CLASSES:
            for i in range(0,len(file)):
                if int(file[i][0:6]) < EPOCH_SPLIT: #assign to train set
                    destination = 'tmp/train/'
                else: #assign to val or test set
                    rand = np.random.uniform(0,1.0)
                    if rand < VAL_TEST_SPLIT: 
                        destination = 'tmp/val/'
                    else:
                        destination = 'tmp/test/'
                copyfile(root + '/' + file[i], destination + root.split('/')[1] + '/' + file[i])

# Set matplotlib interactive mode
plt.ion()  

# NOTE: Per Pytorch documentation, pretrained Pytorch models (resnet18, alexnet, squeezenet, 
# vgg15, densenet, inception, googlenet) assume images of size (3,H,W) where H and W are at 
# least 224.  Images should be loaded in range of [0,1] and normalized using mean = [0.485, 
# 0.456, 0.406] and std = [0.229, 0.224, 0.225]

# Data augmentation and normalization for training (randome parameters will be different at each training epoch)
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(PROB_HORIZ),
        transforms.RandomGrayscale(PROB_GRAY),
        transforms.ColorJitter(COLOR_JITTER[0], COLOR_JITTER[1], COLOR_JITTER[2], COLOR_JITTER[3]),
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
image_datasets = {x: datasets.ImageFolder(os.path.join('tmp', x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}

# Setup device for processing, default is cpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Show examples of training data
# inputs, classes = next(iter(dataloaders['train'])) # Get a batch of training data
# out = torchvision.utils.make_grid(inputs) # Make a grid from batch
# hlp.imshow(out, title=[class_names[x] for x in classes])
# input('press <ENTER> to continue')

# Function for training model
def train_model(model, criterion, optimizer, scheduler, EVAL_FLAG, num_epochs=25):
    since = time.time()
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    print(dataset_sizes)
    class_names = image_datasets['train'].classes
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    phases = ['train', 'val']
    if EVAL_FLAG == 1:
        phases = ['val']

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in phases:
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

def Freeze_Params(m):
    num_layers = 0
    for param in m.parameters(): 
        num_layers += 1
    cnt = 0
    for param in m.parameters():     
        if cnt == NUM_FREEZE:
            break
        print('FREEZING LAYER', cnt+1, 'of', num_layers)
        param.requires_grade = False
        cnt += 1
    return m

# Initialize model and replace last layer w/ new fully connected layer
if PRETRAIN_NET == 'RES':
    model_ft = models.resnet18(pretrained=True)
    if FREEZE_PRIOR == True:
        model_ft = Freeze_Params(model_ft)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, len(CLASSES))
elif PRETRAIN_NET == 'ALEX':
    model_ft = models.alexnet(pretrained=True) 
    if FREEZE_PRIOR == True:
        model_ft = Freeze_Params(model_ft)
    num_ftrs = model_ft.classifier[6].in_features
    model_ft.classifier[6] = nn.Linear(num_ftrs, len(CLASSES))
elif PRETRAIN_NET == 'VGG':
    model_ft = models.vgg19(pretrained=True)
    if FREEZE_PRIOR == True:
        model_ft = Freeze_Params(model_ft)
    num_ftrs = model_ft.classifier[6].in_features
    model_ft.classifier[6] = nn.Linear(num_ftrs, len(CLASSES))

# Continue from last saved model
if CONTINUE_FLAG == 1: 
    print('Loading Saved Model: ', MODEL_PATH)
    model_ft.load_state_dict(torch.load(MODEL_PATH))

# Assign model to device (CPU if local)
model_ft = model_ft.to(device)

# Set up optimize
if OPTIMIZER == 'SGD':
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, momentum=SGD_SETTINGS[1], nesterov=SGD_SETTINGS[0])
elif OPTIMIZER == 'ADAM':
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# Setup learning rate decay schedule (note: LR by a factor of 0.1 every 7 epochs)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=LR_DECAY_EPOCHS, gamma=LR_DECAY_FACTOR)

# Train and return model
criterion = nn.CrossEntropyLoss()

model_ft, best_acc = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, EVAL_FLAG, num_epochs=NUM_EPOCHS)

# Save model
torch.save(model_ft.state_dict(), MODEL_PATH)

# Visualize model
# visualize_model(model_ft)
# input('press <ENTER> to continue')

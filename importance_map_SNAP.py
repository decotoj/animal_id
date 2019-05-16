#Adapted from following
#https://www.kaggle.com/ceshine/pytorch-deep-explainer-mnist-example

import torch, torchvision
from torchvision import datasets, transforms, models
from torch import nn, optim
from torch.nn import functional as F
import os
import numpy as np
import shap
import time

NUMCLASSES = 4
batch_size = 2
device = torch.device('cpu')
DATA_DIR = 'data3'
MODEL_FILE = 'models/model3.pt'

# test_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('mnist_data', train=False, transform=transforms.Compose([
#                        transforms.ToTensor()
#                    ])),
#     batch_size=batch_size, shuffle=True)



data_transforms = {
    'val': transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'vis': transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor()
    ])
}

image_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'val'),data_transforms['val'])
test_loader_2 = torch.utils.data.DataLoader(
    image_dataset,batch_size=batch_size, shuffle=True)

image_dataset_3 = datasets.ImageFolder(os.path.join(DATA_DIR, 'val'),data_transforms['vis'])
test_loader_3 = torch.utils.data.DataLoader(
    image_dataset_3,batch_size=batch_size, shuffle=True)

# Initialize model
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, NUMCLASSES)

model.load_state_dict(torch.load(MODEL_FILE))

# Assign model to device (CPU if local)
model = model.to(device)

# since shuffle=True, this is a random sample of test data
batch = next(iter(test_loader_2))
images, _ = batch


# ####################################
batch2 = next(iter(test_loader_3))
images2, _ = batch2
test_images2 = images2[1:2]
# #####################################

background = images[:1]
test_images = images[1:2]

e = shap.DeepExplainer(model, background)
shap_values = e.shap_values(test_images)

shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
test_numpy = np.swapaxes(np.swapaxes(test_images2.numpy(), 1, -1), 1, 2)

#TEST1234
test_numpy = test_numpy[:,:,:,0] #Plotting works if only one color channel is used 

#Causes 3 copies of image in color channels to be plotted
# shap_numpy = [np.swapaxes(np.swapaxes(np.swapaxes(s, 1, -1), 1, 2), 0, -1) for s in shap_values]
# test_numpy = np.swapaxes(np.swapaxes(np.swapaxes(test_images2.numpy(), 1, -1), 1, 2), 0, -1)

print('background', type(background), background.shape)
print('test_images', type(test_images), test_images.shape)
print('shap_numpy', type(shap_numpy), len(shap_numpy), shap_numpy[0].shape)
print('test_numpy', type(test_numpy), test_numpy.shape)


# X,y = shap.datasets.imagenet50()
# to_explain = X[[39]] #Example onlien works with [1,224,224,3] input for 'test_numpy'
# print('test', to_explain.shape)

#plot the feature attributions
#shap.image_plot(shap_numpy, -test_numpy)

shap.image_plot(shap_numpy, np.swapaxes(np.swapaxes(images2[1:2].numpy(), 1, -1), 1, 2))

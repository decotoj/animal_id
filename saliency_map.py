import torch
import torchvision
import torchvision.transforms as T
import random
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
import matplotlib.pyplot as plt
#from cs231n.image_utils import SQUEEZENET_MEAN, SQUEEZENET_STD
from PIL import Image
from torchvision import datasets, models, transforms
import os
import torch.nn as nn

#matplotlib inline
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# Constants
NUMCLASSES = 4 #Number of Classification Classes
NUM_EPOCHS = 2 #number of training epochs (baseline = 25)
MODEL_PATH = 'models/model3.pt' #save path for model file
CONTINUE_FLAG = 1 #1=load last saved model file and continue training else start fresh from pre-trained pytorch model
DATA_DIR = 'data3'
BATCH_SIZE = 4

# Setup device for processing, default is cpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Initialize model
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, NUMCLASSES)

#Load Model File
model.load_state_dict(torch.load(MODEL_PATH))

# Assign model to device (CPU if local)
model = model.to(device)

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
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

# def preprocess(img, size=224):
#     transform = T.Compose([
#         T.Resize(size),
#         T.ToTensor(),
#         T.Normalize(mean=SQUEEZENET_MEAN.tolist(),
#                     std=SQUEEZENET_STD.tolist()),
#         T.Lambda(lambda x: x[None]),
#     ])
#     return transform(img)

def compute_saliency_maps(X, y, model):
    """
    Compute a class saliency map using the model for images X and labels y.

    Input:
    - X: Input images; Tensor of shape (N, 3, H, W)
    - y: Labels for X; LongTensor of shape (N,)
    - model: A pretrained CNN that will be used to compute the saliency map.

    Returns:
    - saliency: A Tensor of shape (N, H, W) giving the saliency maps for the input
    images.
    """
    # Make sure the model is in "test" mode
    model.eval()
    
    # Make input tensor require gradient
    X.requires_grad_()
    
    saliency = None
    ##############################################################################
    # TODO: Implement this function. Perform a forward and backward pass through #
    # the model to compute the gradient of the correct class score with respect  #
    # to each input image. You first want to compute the loss over the correct   #
    # scores (we'll combine losses across a batch by summing), and then compute  #
    # the gradients with a backward pass.                                        #
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    out = model(X)  #forward pass    
    score = out.gather(1, y.view(-1, 1)).squeeze() #score for truth class
    score.backward(torch.ones(score.shape)) #backward pass
    grad = X.grad #get gradients
    grad = grad.abs() #absolute value of gradients
    saliency,_ = torch.max(grad, dim=1) #max across input channels

    #NOTE: Explanation of why argument is needed to be passed to 'torch.backward()'
    #https://discuss.pytorch.org/t/loss-backward-raises-error-grad-can-be-implicitly-created-only-for-scalar-outputs/12152
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                             END OF YOUR CODE                               #
    ##############################################################################
    return saliency

def show_saliency_maps(X, y):

    print(type(X))

    # Convert X and y from numpy arrays to Torch Tensors
    X_tensor = X#torch.cat([preprocess(Image.fromarray(x)) for x in X], dim=0)
    y_tensor = y#torch.LongTensor(y)

    # Compute saliency maps for images in X
    saliency = compute_saliency_maps(X_tensor, y_tensor, model)

    # Convert the saliency map from Torch Tensor to numpy array and show images
    # and saliency maps together.
    saliency = saliency.numpy()
    N = X.shape[0]
    for i in range(N):
        plt.subplot(2, N, i + 1)
        plt.imshow(X[i].detach().numpy().swapaxes(0,2))
        plt.axis('off')
        plt.title(class_names[y[i]])
        plt.subplot(2, N, N + i + 1)
        plt.imshow(saliency[i], cmap=plt.cm.hot)
        plt.axis('off')
        plt.gcf().set_size_inches(12, 5)
    plt.show()

for inputs, labels in dataloaders['val']:

    print('inputs', inputs.shape)
    show_saliency_maps(inputs, labels)
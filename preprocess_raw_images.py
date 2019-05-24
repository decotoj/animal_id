import PIL
from PIL import Image, ExifTags
import os
import numpy as np

#Processes an input folder of raw images (organized by class) and resizes all images, then randomely 
#assigns images to either the train, validation, or test set

#Relative path to folder with raw data
#Data assumed to be organized into subfolder by class (ex: 'data/raw/001', 'data/raw/002')
dataDirectory = "/media/jake/My Passport2/Pictures/2019-06-01_Cow_Pictures"
train_val_test_Directory = ["data/all", "data/train", "data/val", "data/test"]
train_val_test_ratios = [1.0, 0.0, 0.0, 0.0]
basewidth = 400 #Width for resized images (aspect ratio will be kept the same)

#Get List of All Image Files in Raw Data Directory
files = []
for root, dirs, file in os.walk(dataDirectory):
    for i in range(0,len(file)):
        if '.JPG' in file[i] or '.jpg' in file[i]:
            files.append(root + '/' + file[i])
print('Number of Images:', len(files))

#Purge Current Train/Val/Test Directories
for path in train_val_test_Directory:
    for root, dirs, file in os.walk(path):  
        #files = files + file
        for i in range(0,len(file)):
            os.remove(root + '/' + file[i])

#Step Through and Resize All of The Images 
for i in range(len(files)):
    img = Image.open(files[i])

    c = Image.open(files[i])._getexif()[36867]
    c = c[2:4] + c[5:7] + c[8:10]

    wpercent = (basewidth / float(img.size[0]))
    if '.JPG' in files[i]:
        hsize = int((float(img.size[1]) * float(wpercent)))
    else:
        hsize = 300
    img = img.resize((basewidth, hsize), PIL.Image.ANTIALIAS)

    
    #img = img.rotate(180) #rotate to keep original orientation

    rand = np.random.uniform(0,1.0)
    
    if rand<train_val_test_ratios[0]:
        pth = train_val_test_Directory[0]
    elif rand<(train_val_test_ratios[0] + train_val_test_ratios[1]):
        pth = train_val_test_Directory[1]
    else:
        pth = train_val_test_Directory[2]
    print(i, 'of', len(files), pth)

    tmp = files[i].replace(dataDirectory,'')
    tmp = tmp.split('/')
    out = pth + '/' + tmp[1] + '/' + c + '_' + tmp[2]
    img.save(out)
print('Done Resizing and Assigning Images to Train/Val/Test Sets')
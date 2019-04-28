import PIL
from PIL import Image
from PIL import ImageEnhance
import os
import numpy as np

def crop_image(img, min_pct=0.5):
    # Select size of crop, at least 50% of original image in both dimensions
    new_width = np.random.choice(range(int(min_pct*img.size[0]), img.size[0]))
    new_height = np.random.choice(range(int(min_pct*img.size[1]), img.size[1]))
    top_left_corner = (np.random.choice(range(img.size[0]-new_width)), np.random.choice(range(img.size[1]-new_height)))
    img = img.crop(box=(top_left_corner[0], top_left_corner[1], top_left_corner[0]+new_width, top_left_corner[1]+new_height))
    return img

def cutout_image(img, cutout_size=(75,75)):
    top_left_corner = (np.random.choice(range(img.size[0]-cutout_size[0])), np.random.choice(range(img.size[1]-cutout_size[1])))
    for i in range(top_left_corner[0], top_left_corner[0]+cutout_size[0]):
        for j in range(top_left_corner[1], top_left_corner[1]+cutout_size[1]):
            img.putpixel(xy=(i, j), value=(0,0,0))
    return img

def resize_image(img, basewidth):
    wpercent = (basewidth / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((basewidth, hsize), PIL.Image.ANTIALIAS)
    img = img.rotate(180) #rotate to keep original orientation
    return img

# Experimentally, 0.5-1.5 seems like a reasonable parameter range for these functions
def contrast_image(img, min_val=0.5):
    contraster = ImageEnhance.Contrast(img)
    img = contraster.enhance(min_val+np.random.rand())
    return img

def brighten_image(img, min_val=0.5):
    brightener = ImageEnhance.Brightness(img)
    img = brightener.enhance(min_val+np.random.rand())
    return img

def sharpen_image(img, min_val=0.5):
    sharpener = ImageEnhance.Brightness(img)
    img = sharpener.enhance(min_val+np.random.rand())
    return img

def transform_all_images():
    #Processes an input folder of raw images (organized by class) and resizes all images, then randomly 
    #assigns images to either the train, validation, or test set

    #Relative path to folder with raw data
    #Data assumed to be organized into subfolder by class (ex: 'data/raw/001', 'data/raw/002')
    dataDirectory = "data/raw" 
    train_val_test_Directory = ["data/train", "data/val", "data/test"]
    train_val_test_ratios = [0.8, 0.1, 0.1]
    BASEWIDTH = 400 #Width for resized images (aspect ratio will be kept the same)

    #Get List of All Image Files in Raw Data Directory
    files = []
    for root, dirs, file in os.walk(dataDirectory):  
        for i in range(0,len(file)):
            files.append(root + '/' + file[i])
    print('Number of Images:', len(files))

    #Purge Current Train/Val/Test Directories
    for path in train_val_test_Directory:
        for root, dirs, file in os.walk(path):  
            for i in range(0,len(file)):
                os.remove(root + '/' + file[i])

    #Step Through and Resize All of The Images 
    for i in range(len(files)):
        basewidth = BASEWIDTH
        img = Image.open(files[i])
        img = resize_image(img, basewidth)
        
        rand = np.random.uniform(0,1.0)
        
        if rand<train_val_test_ratios[0]:
            pth = train_val_test_Directory[0]
        elif rand<(train_val_test_ratios[0] + train_val_test_ratios[1]):
            pth = train_val_test_Directory[1]
        else:
            pth = train_val_test_Directory[2]
        print(i, 'of', len(files), pth)
        img.save(files[i].replace(dataDirectory,pth))
    print('Done Transforming and Assigning Images to Train/Val/Test Sets')

img = Image.open(open(os.getcwd()+"/cow.jpg", "rb"))
img2 = contrast_image(img)
img2.show()
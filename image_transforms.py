import PIL
from PIL import Image
from PIL import ImageEnhance
import os
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

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

def flip_image(img):
    return img.transpose(Image.FLIP_LEFT_RIGHT)

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

def get_eigenvalues(dataDirectory="data/train"):
    #Get List of All Image Files in Raw Data Directory
    files = []
    for root, dirs, file in os.walk(dataDirectory):  
        for i in range(0,len(file)):
            files.append(root + '/' + file[i])
    
    X = np.asarray(Image.open(files[0])).reshape((360000,1))
    for file_i in range(1, len(files)):
        X = np.concatenate((X, np.asarray(Image.open(files[file_i])).reshape((360000,1))), axis=1)

    X = X.astype(float)

    X -= np.mean(X, axis=0)
    cov = np.cov(X, rowvar=False)
    evals, evecs = linalg.eigh(cov)

    idx = np.argsort(evals)[::-1]
    evecs = evecs[:,idx]
    evals = evals[idx]

    return evals

def plot_cumsum(evals, filename, alpha=0.9):
    x = range(len(evals))
    tot = np.sum(evals)
    y = [float(i)/tot for i in np.cumsum(evals)]
    pcs = np.argmax([int(j >= alpha) for j in y])
    print(y)
    plt.plot(x, y)
    plt.ylabel('Percentage of Variance Explained')
    plt.xlabel('Number of Principal Components')
    plt.title('Dimensionality of Training Dataset')
    plt.savefig(filename)
    return pcs, x, y
    

def augment_train_images(operations, basewidth=400):
    dataDirectory = "data/train" 

    #Get List of All Image Files in Raw Data Directory
    files = []
    for root, dirs, file in os.walk(dataDirectory):  
        for i in range(0,len(file)):
            files.append(root + '/' + file[i])

    for file_i in files:
        if file_i.count('_') > 1:
            pass
        img = Image.open(file_i)
        try:
            ext = file_i.index('.JPG')
        except:
            ext = file_i.index('.jpg')
        if 'BRIGHT' in operations or operations == 'ALL':
            bright = resize_image(brighten_image(img), basewidth)
            bright.save(file_i[:ext]+'_BRIGHT.JPG', 'JPEG')
        if 'SHARP' in operations or operations == 'ALL':
            sharp = resize_image(sharpen_image(img), basewidth)
            sharp.save(file_i[:ext]+'_SHARP.JPG', 'JPEG')
        if 'FLIP' in operations or operations == 'ALL':
            flipped = resize_image(flip_image(img), basewidth)
            flipped.save(file_i[:ext]+'_FLIP.JPG', 'JPEG')
        if 'CUTOUT' in operations or operations == 'ALL':
            cutout = resize_image(cutout_image(img), basewidth)
            cutout.save(file_i[:ext]+'_CUTOUT.JPG', 'JPEG')
        if 'CROP' in operations:
            cropped = resize_image(crop_image(img), basewidth)
            cropped.save(file_i[:ext]+'_CROP.JPG', 'JPEG')
        if 'CONTRAST' in operations or operations == 'ALL':
            contrast = resize_image(contrast_image(img), basewidth)
            contrast.save(file_i[:ext]+'_CONTRAST.JPG', 'JPEG')

def split_train_val_test():
    #Processes an input folder of raw images (organized by class) and resizes all images, then randomly 
    #assigns images to either the train, validation, or test set

    #Relative path to folder with raw data
    #Data assumed to be organized into subfolder by class (ex: 'data/raw/001', 'data/raw/002')
    dataDirectory = "images" 
    train_val_test_Directory = ["data/train", "data/val", "data/test"]
    train_val_test_ratios = [0.8, 0.1, 0.1]
    BASEWIDTH = 400 #Width for resized images (aspect ratio will be kept the same)

    #Get List of All Image Files in Raw Data Directory
    files = []
    for root, dirs, file in os.walk("data/train"):  
        for i in range(0,len(file)):
            files.append(root + '/' + file[i])
            print(Image.open(files[i]).size)
    print('Number of Images:', len(files))

    #Purge Current Train/Val/Test Directories
    for path in train_val_test_Directory:
        for root, dirs, file in os.walk(path):  
            for i in range(0,len(file)):
                os.remove(root + '/' + file[i])

    #Step Through and Resize All of The Images 
    for i in range(len(files)):
        img = Image.open(files[i])
        img = resize_image(img, BASEWIDTH)
        
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


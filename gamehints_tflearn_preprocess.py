import cv2                 # working with, mainly resizing, images
import numpy as np         # dealing with arrays
import os                  # dealing with directories
from random import shuffle # mixing up or currently ordered data that might lead our network astray in training.
from tqdm import tqdm      # a nice pretty percentage bar for tasks. Thanks to viewer Daniel BA1/4hler for this suggestion



TRAIN_DIR = './Part1_The_Forest' # where ever we have extracted our files to.
TEST_DIR = 'X:/Kaggle_Data/dogs_vs_cats/test/test'
IMG_SIZE = 28
LR = 1e-3 #learning rate

MODEL_NAME = 'dogsvscats-{}-{}.model'.format(LR, '2conv-basic') # just so we remember which saved model is which, sizes must match

'''
Now, our first order of business is to convert the images and labels to array information that we can pass through our network. To do this, we'll need a helper function to convert the image name to an array.

Our images are labeled like "cat.1" or "dog.3" and so on, so we can just split out the dog/cat, and then convert to an array like so:
'''

def label_img(img):
    word_label = ''
    # conversion to one-hot array [cat,dog]
    #                            [much cat, no dog]
    if word_label == 'Checkpoint_1': return [1,0,0,0,0,0]
    #                             [no cat, very doggo]
    elif word_label == 'Checkpoint_2': return [0,1,0,0,0,0]

    elif word_label == 'Checkpoint_3': return [0,0,1,0,0,0]

    elif word_label == 'Checkpoint_4': return [0,0,0,1,0,0]

    elif word_label == 'Checkpoint_5': return [0,0,0,0,1,0]

    elif word_label == 'Checkpoint_6': return [0,0,0,0,0,1]


#Now, we can build another function to fully process the training images and their labels into arrays
#converts the data for us into array data of the image and its label.
def create_train_data():
    training_data = []
    index = 1
    for img in tqdm(os.listdir(os.path.join(TRAIN_DIR + "/Checkpoint_" + str(index)))):
        label = label_img("Checkpoint_" + str(index))
        path = os.path.join(TRAIN_DIR + "/Checkpoint_" + str(index),img)
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        training_data.append([np.array(img),np.array(label)])
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    
    return training_data


def process_test_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR,img)
        img_num = img.split('.')[0]
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        testing_data.append([np.array(img), img_num])
        
    shuffle(testing_data)
    np.save('test_data.npy', testing_data)

    return testing_data


train_data = create_train_data()
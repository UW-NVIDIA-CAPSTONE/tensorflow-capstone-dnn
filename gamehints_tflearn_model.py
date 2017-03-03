import cv2                 # working with, mainly resizing, images
import numpy as np         # dealing with arrays
import os                  # dealing with directories
from random import shuffle # mixing up or currently ordered data that might lead our network astray in training.
from tqdm import tqdm      # a nice pretty percentage bar for tasks. Thanks to viewer Daniel BA1/4hler for this suggestion

TRAIN_DIR = './Part1_The_Forest' # where ever we have extracted our files to.
TEST_DIR = './Validation_Set'
IMG_SIZE = 28
LR = 1e-3 #learning rate
MODEL_NAME = 'insideforest-{}-{}.model'.format(LR, '2conv-basic') # just so we remember which saved model is which, sizes must match

'''
Now, our first order of business is to convert the images and labels to array information that we can pass through our network. To do this, we'll need a helper function to convert the image name to an array.

Our images are labeled like "cat.1" or "dog.3" and so on, so we can just split out the dog/cat, and then convert to an array like so:
'''
'''
def label_img(img):
    word_label = ''
    # conversion to one-hot array [cat,dog]
    #                            [much cat, no dog]
    if word_label == 'Checkpoint_1': return [1,0,0,0,0,0]
    #                             [no cat, very doggo]
    elif word_label == 'Checkpoint_2': return [0,1,0,0,0,0]
    '''
'''
    elif word_label == 'Checkpoint_3': return [0,0,1,0,0,0]

    elif word_label == 'Checkpoint_4': return [0,0,0,1,0,0]

    elif word_label == 'Checkpoint_5': return [0,0,0,0,1,0]

    elif word_label == 'Checkpoint_6': return [0,0,0,0,0,1]
'''

#Now, we can build another function to fully process the training images and their labels into arrays
#converts the data for us into array data of the image and its label.

'''
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
'''


#tflearn model, higher level abstraction layer library
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn.data_utils import image_preloader

import tensorflow as tf

tf.reset_default_graph()
convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 128, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 6, activation='softmax')

convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')

'''
if os.path.exists('{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print('model loaded!')
'''
'''
train = train_data[:-500]
test = train_data[-500:]

X = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
Y = [i[1] for i in train]

test_x = np.array([i[0] for i in test]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
test_y = [i[1] for i in test]
'''

X, Y = image_preloader(TRAIN_DIR, image_shape=(28, 28),   mode='folder', categorical_labels=True,   normalize=True)
test_x, test_y = image_preloader(TEST_DIR, image_shape=(28, 28),   mode='folder', categorical_labels=True,   normalize=True)
#model.fit(X , Y, n_epoch=3, validation_set=({'input': test_x}, {'targets': test_y}), snapshot_step=500, show_metric=True, run_id=MODEL_NAME)
model.fit({'input': X}, {'targets': Y}, n_epoch=3, validation_set=({'input': test_x}, {'targets': test_y}), snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

model.save(MODEL_NAME)
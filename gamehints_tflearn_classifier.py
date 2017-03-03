#prediction, or evaluation 
# Save a model

import cv2                 # working with, mainly resizing, images
import numpy as np         # dealing with arrays
import os                  # dealing with directories
from random import shuffle # mixing up or currently ordered data that might lead our network astray in training.
from tqdm import tqdm 
LR = 1e-3 #learning rate


MODEL_NAME = 'insideforest-{}-{}.model'.format(LR, '2conv-basic')

INPUT_DIR = './Input' # where ever we have extracted our files to.

#X, _ = image_preloader(INPUT_DIR, image_shape=(28, 28),   mode='folder', categorical_labels=True,   normalize=True)

if os.path.exists('{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print('model loaded!')
    #model.predict(X)
else:
	print('no model found.')


# Load a model

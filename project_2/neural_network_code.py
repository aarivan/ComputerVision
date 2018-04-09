# import the necessary packages
import numpy as np
import random
import cv2
import glob
import os
# import argparse 


# from pyimagesearch.lenet import LeNet
from imutils import paths


# import the necessary packages for tensorflow and keras
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from keras import backend as K


# set the matplotlib backend so figures can be saved in the background
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")


# import sklearn packages
from sklearn.model_selection import train_test_split


# import tensorflow as tf
import keras.backend.tensorflow_backend as ktf 


def get_session(gpu_fraction=0.333):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction,
                                allow_growth=True)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


ktf.set_session(get_session())


# Dataset Paths defined
DATASET_DIR = "../Datasets/CelebA/Data/img_align_celeba/"
PATH = DATASET_DIR + "*.jpg"
FACES_DATASET = "../Faces_dataset/"
NON_FACES_DATASET = "../Non_Faces_dataset/"
FACES_DATASET_TRAIN = "../Faces_dataset_train/"
NON_FACES_DATASET_TRAIN = "../Non_Faces_dataset_train/"
FACES_DATASET_TEST = "../Faces_dataset_test/"
NON_FACES_DATASET_TEST = "../Non_Faces_dataset_test/"

# Initialize the number of epochs to train for, initial learning rate, and batch size
EPOCHS = 1
INIT_LR = 1e-3
BS = 32

# print(os.getcwd())

# Iterate through the faces dataset and crop the non-face images
# Extract image datasets and split them into train and test data (20k, 5k respectively)

''' 
            <<--------- UNCOMMENT THIS BLOCK OF CODE --------->>

cv_img = []
for img in glob.glob(PATH):
    # img = "../Datasets/CelebA/Data/img_align_celeba\000001.jpg" --- img[41:]
    if len(cv_img) < 25000:
        image = cv2.imread(img)
        face_image = cv2.resize(image, (60, 60))
        non_face_image = image[0 : 60, 0 : 60]
        cv2.imwrite(FACES_DATASET + img[41:], face_image)
        cv2.imwrite(NON_FACES_DATASET + img[41:], non_face_image)
        cv_img.append(image)
    else:
        break

print ("completed: len of cv_img:", len(cv_img))
'''


# Initializing the data and labels
# These lists will be responsible for storing our the images we load from disk along with their respective class labels
print("[INFO] loading images...")

image_data = []
labels = []

# grab the image paths and randomly shuffle them
# we grab the paths to our input images followed by shuffling them
face_image_paths = sorted(list(paths.list_images(FACES_DATASET_TRAIN)))
non_face_image_paths = sorted(list(paths.list_images(NON_FACES_DATASET_TRAIN)))

train_image_paths = face_image_paths[:10000] + non_face_image_paths[:10000]
random.seed(42)
random.shuffle(train_image_paths)


# Pre-process the images
# loop over the input images
for img in train_image_paths:
    # load the image, pre-process it, and store it in the data list
    image = cv2.imread(img)
    image = img_to_array(image)
    image_data.append(image)
    
    # extract the class label from the image path and update the
	# labels list
    label = img.split('/')[-2]
    label = 1 if label == "Faces_dataset_train" else 0
    labels.append(label)


# Pre-processing the data - making it zero-centered
data = np.array(image_data, dtype = np.float32)
data -= np.mean(data, axis = 0)
data /= np.std(data, axis = 0)
labels = np.array(labels)

# # scale the raw pixel intensities to the range [0, 1]
# data = np.array(data, dtype="float") / 255.0
# labels = np.array(labels)
 
# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size = 0.25, random_state = 42)
 
# convert the labels from integers to vectors
trainY = to_categorical(trainY, num_classes=2)
testY = to_categorical(testY, num_classes=2)


# initialize the model
print("[INFO] compiling model...")
width = 60 
height= 60
depth = 3
classes = 2
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)





# Hyper-parameters
# 1. Batch size - 1, 10, 100, 1000
# 2. Learning rate - 0.001, 0.01 ,  0.1, 1 
# 3. Activation - relu, leaky relu, sigmoid, tanh
# 4. Decay rate - << only for adam, adagrad - bias 1 and bias 2, momentum parameter >>
# 5. Optimization - adam, sgd, momentum, rms-prop, ada-grad 

# Building the neural network
model = Sequential()
input_shape = (height, width, depth)

tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph = True, write_images = True)

# if we are using "channels first", update the input shape
if K.image_data_format() == "channels_first":
    input_shape = (depth, height, width)
# else:
#     input_shape = (img_rows, img_cols, 1)


# first set of CONV => RELU => POOL layers
# 20 convolution filters, each of which are 5×5.
model.add(Conv2D(32, (5, 5), padding="same", input_shape=input_shape))
# apply a ReLU activation function
model.add(Activation("relu"))
# followed by 2×2 max-pooling in both the x and y direction with a stride of two
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))


# second set of CONV => RELU => POOL layers 
# 50 convolutional filters - common to see the number of CONV  filters learned increase 
# the deeper we go in the network architecture
model.add(Conv2D(64, (5, 5), padding="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))


# first (and only) set of FC => RELU layers
# take the output of the preceding MaxPooling2D  layer and 
# flatten it into a single vector. 
# This operation allows us to apply our dense/fully-connected layers
model.add(Flatten())
# Our fully-connected layer contains 500 nodes which we then pass through another nonlinear ReLU activation.
model.add(Dense(500), kernel_regularizer = regularizers.l2(0))
model.add(Activation("relu"))


# softmax classifier
# defines another fully-connected layer, 
# but this one is special — the number of nodes is equal to the number of classes  
# (i.e., the classes we want to recognize).
model.add(Dense(classes))
# fed into our softmax classifier which will yield the probability for each class.
model.add(Activation("softmax"))


model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
 

H = model.fit(trainX, trainY,
          batch_size=BS,
          epochs=EPOCHS,
          verbose=1,
          validation_data=(testX, testY),
          callbacks=[tbCallBack])

# # construct the image generator for data augmentation
# aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
# 	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
# 	horizontal_flip=True, fill_mode="nearest")



# # train the network
# print("[INFO] training network...")
# H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
# 	validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
# 	epochs=EPOCHS, verbose=1)
 
# save the model to disk
print("[INFO] serializing network...")
model.save("saved_model.h5")


# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on Face/Non-Face Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot") 


from distutils.command.config import config
import torch
import numpy as np
from torchvision import datasets, transforms
import torchvision
import torch.optim as optim
import pandas as pd
from matplotlib import pyplot as plt
import os
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Activation, Convolution2D,MaxPooling2D,BatchNormalization, Flatten, Dense, Dropout
from keras.layers import LeakyReLU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
 
gpu_options = tf.compat.v1.GPUOptions(allow_growth = True)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options = gpu_options))



# Load the data using np.load
data = np.load('CNN/training_images.npz', allow_pickle=True)

# Extract the images
images = data['images']
# and the data points
pts = data['points']

# pts = pd.read_csv

data_train = np.load("CNN/test_images.npz")
data_example = np.load("CNN/examples.npz")

images_example = data_example["images"]
images_train = data_train["images"]

array_list = []
array_list_train = []
array_list_example = []
for i in range(len(images)-1):
    img = np.mean(images[i],axis=-1)
    array_list.append(img)


x_train = np.array(array_list,dtype="float")
x_train = x_train.reshape(-1,244,244,1)

for i in range(len(images_train)+1):
    img = np.mean(images_train[i],axis=-1)
    array_list_train.append(img)

x_test = np.array(array_list_train,dtype="float")
x_test = x_test.reshape(-1,244,244,1)

print(len(images_example))
for i in range(len(images_example)+1):
    img = np.mean(images_example[i],axis=-1)
    array_list_example.append(img)

x_example = np.array(array_list_example,dtype="float")
x_example = x_example.reshape(-1,244,244,1)




keypoints_df = pd.read_csv("CNN/keypoint.csv")
y_train = np.array(keypoints_df ,dtype="float")
#def the algorithmic to show the train image
def visualizeWithNoKeypoints(index):
    plt.imshow(x_train[index].reshape(244,244),cmap='gray')
def visualizeWithKeypoints(index):
    plt.imshow(x_train[index].reshape(244,244),cmap='gray')
    for i in range(1,47,2):
        plt.plot(y_train[0][i-1],y_train[0][i],'+r')

# fig = plt.figure(figsize=(8,4))
# fig.subplots_adjust(left=0,right=1,bottom=0,top=1,hspace=0.05,wspace=0.05)
# plt.subplot(1,2,1)
# visualizeWithNoKeypoints(1)
# plt.subplot(1,2,2)
# visualizeWithKeypoints(1)
# plt.show()
 
# build the CNN 
model = Sequential()
model.add(Convolution2D(3,(5,5),padding='valid',use_bias=False, input_shape=(244,244,1)))# the first layer, use 3 5*5 operators
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))# make the 244*244 image to 122*122 (reduce the data size)


model.add(Convolution2D(96,(5,5),padding='valid',use_bias = False))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
 
model.add(Convolution2D(256,(3,3),padding='valid',use_bias = False))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization())


model.add(Convolution2D(384, (3,3), padding='valid', use_bias=False))# use so many operators to make the model get better accuracy rate
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(256, (4,4), padding='valid', use_bias=False))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())
 
model.add(Flatten())

model.add(Dense(256,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(84))
 
# show the progress before that could know if need to change the operators size or something other.
model.summary()

# delte the useless data 
model.compile(optimizer='adam',loss='mean_squared_error',metrics=['mae','acc'],run_eagerly=True)

model.fit(x_train,y_train,batch_size=64,epochs=1000,validation_split=0.7)

pred = model.predict(x_test)
pred_train = model.predict(x_train)
pred_example = model.predict(x_example)


# save the data as csv
def save_as_csv(points, location = '.'):
    """
    Save the points out as a .csv file
    :param points: numpy array of shape (no_test_images, no_points, 2) to be saved
    :param location: Directory to save results.csv in. Default to current working directory
    """
    assert points.shape[0]==554, 'wrong number of image points, should be 554 test images'
    assert np.prod(points.shape[1:])==2*42, 'wrong number of points provided. There should be 42 points with 2 values (x,y) per point'
    np.savetxt(location + '/pre_results_yzf.csv', np.reshape(points, (points.shape[0], -1)), delimiter=',')

save_as_csv(pred,location="C:\Report\CV")
def save_as_csv_train(points, location = '.'):
    """
    Save the points out as a .csv file
    :param points: numpy array of shape (no_test_images, no_points, 2) to be saved
    :param location: Directory to save results.csv in. Default to current working directory
    """
    assert points.shape[0]==2810, 'wrong number of image points, should be 554 test images'
    assert np.prod(points.shape[1:])==2*42, 'wrong number of points provided. There should be 42 points with 2 values (x,y) per point'
    np.savetxt(location + '/keypoints_pre.csv', np.reshape(points, (points.shape[0], -1)), delimiter=',')
save_as_csv_train(pred_train,location="C:\Report\CV")

def save_as_csv_example(points, location = '.'):
    """
    Save the points out as a .csv file
    :param points: numpy array of shape (no_test_images, no_points, 2) to be saved
    :param location: Directory to save results.csv in. Default to current working directory
    """
    assert points.shape[0]==6, 'wrong number of image points, should be 554 test images'
    assert np.prod(points.shape[1:])==2*42, 'wrong number of points provided. There should be 42 points with 2 values (x,y) per point'
    np.savetxt(location + '/pre_example.csv', np.reshape(points, (points.shape[0], -1)), delimiter=',')
save_as_csv_example(pred_example ,location="C:\Report\CV")

# print(len(pred))

# make sure the csv size is 544
print(len(pred))



 

 

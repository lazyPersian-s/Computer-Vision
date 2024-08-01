from distutils.command.config import config
import torch
import cv2
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

pre_example = pd.read_csv("CNN\pre_example.csv",header=None) #the keypoints can be saved in the training_images.npz as .cvs
y_example = np.array(pre_example ,dtype="float")

example = np.load("CNN\examples.npz")
images_example = example["images"]

pic_num = 5
array_list = []

for i in range(len(images_example)):
    img = np.mean(images_example[i],axis=-1)
    array_list.append(img)
print(len(array_list))

x_example = np.array(array_list,dtype="float")
x_example = x_example.reshape(-1,244,244,1)

#show example points
def visualizeWithNoKeypoints(index):
    plt.imshow(x_example[index].reshape(244,244),cmap='gray')
def visualizeWithKeypoints(index):
    plt.imshow(x_example[index].reshape(244,244),cmap='gray')
    for i in range(1,83,2):
        plt.plot(y_example[pic_num][i-1],y_example[pic_num][i],'+r')


fig = plt.figure(figsize=(8,4))
fig.subplots_adjust(left=0,right=1,bottom=0,top=1,hspace=0.05,wspace=0.05)
plt.subplot(1,2,1)
visualizeWithNoKeypoints(pic_num )
plt.subplot(1,2,2)
visualizeWithKeypoints(pic_num)
plt.show()
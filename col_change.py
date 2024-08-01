from distutils.command.config import config
from tkinter import Image
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

data_train = np.load("CNN/test_images.npz")
img = data_train["images"]
img_num = 4
# print(img.shape)
img_single = img[img_num ]

keypoints_pre = pd.read_csv("CNN/pre_results.csv")
img_dst= np.array(keypoints_pre ,dtype="int")
img_dst = img_dst[img_num]
img_dst = np.reshape(img_dst,(42,2))
# print(img_dst)
color = [255,23,140]



for x in range(img_dst[7][0],img_dst[13][0]):
    for y in range(img_dst[8][1],img_dst[19][1]):
        img_single[y,x] = np.mean(img_single[y,x],axis=-1)

for x in range(img_dst[10][0],img_dst[12][0]):
    for y in range(img_dst[11][1],img_dst[20][1]):
        img_single[y,x] = np.mean(img_single[y,x],axis=-1)

for x in range(img_dst[22][0],img_dst[33][0]):
    for y in range(img_dst[23][1],img_dst[33][1]):
        img_single[y,x] = np.mean(img_single[y,x],axis=-1)

for x in range(img_dst[33][0],img_dst[30][0]):
    for y in range(img_dst[24][1],img_dst[31][1]):
        img_single[y,x] = np.mean(img_single[y,x],axis=-1)

for x in range(img_dst[30][0],img_dst[38][0]):
    for y in range(img_dst[27][1],img_dst[29][1]):
        img_single[y,x] = np.mean(img_single[y,x],axis=-1)


plt.imshow(img_single)
plt.show()

plt.show()





# coding: utf-8

# Importing necessary packages:

# In[1]:

import os
import csv
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, BatchNormalization, Dropout
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam
from keras.activations import relu
from sklearn.model_selection import train_test_split
import sklearn


# Crating the model architecture:

# In[2]:

model = Sequential()
model.add(Lambda(lambda x:x/255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25), (1,1))))
model.add(BatchNormalization())
model.add(Conv2D(24,5,5,subsample=(2,2), border_mode='same', activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(36,5,5,subsample=(2,2), border_mode='same', activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(48,5,5,subsample=(2,2), border_mode='same', activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(64,3,3, border_mode='same', activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(64,3,3, border_mode='same', activation='relu'))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(100))
model.add(BatchNormalization())
model.add(Dense(50))
model.add(BatchNormalization())
model.add(Dense(10))
model.add(BatchNormalization())
model.add(Dense(1))
model.summary()


# Train the model on the dataset with extra collected [data](https://drive.google.com/open?id=0B7YmHsPqymnmRy1rajdTSW9tTnc):

# In[5]:


samples = []
with open('./data2/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
samples=samples[1:]
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def generator(samples, batch_size=64, path):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        np.random.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            car_images = []
            steering_angles = []
            for batch_sample in batch_samples: 
                img_center = (np.asarray(Image.open(path + batch_sample[0].split('\\')[-1].split('/')[-1])))
                img_left = (np.asarray(Image.open(path + batch_sample[1].split('\\')[-1].split('/')[-1])))
                img_right = (np.asarray(Image.open(path + batch_sample[2].split('\\')[-1].split('/')[-1])))
                correction = 0.05
                steering_center = float(batch_sample[3])
                steering_left = steering_center + correction
                steering_right = steering_center - correction
                steering_angles.append(steering_center)
                steering_angles.append(steering_left)
                steering_angles.append(steering_right)
                car_images.append(img_center)
                car_images.append(img_left)
                car_images.append(img_right)
            # trim image to only see section with road
            X_train = np.array(car_images)
            y_train = np.array(steering_angles).reshape(len(steering_angles),1)
            yield sklearn.utils.shuffle(X_train, y_train)
            
train_generator = generator(train_samples, batch_size=64, './data2/IMG/')
validation_generator = generator(validation_samples, batch_size=64, './data2/IMG/')
model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch=                    (3*len(train_samples)), validation_data=validation_generator,                    nb_val_samples=(3*len(validation_samples)), nb_epoch=6)
model.save('model5.h5')


# Train the model on the primary provide [dataset](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip):

# In[4]:


samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
samples=samples[1:]
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

train_generator = generator(train_samples, batch_size=64, './data/IMG/')
validation_generator = generator(validation_samples, batch_size=64, './data/IMG/')
model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch=                    (3*len(train_samples)), validation_data=validation_generator,                    nb_val_samples=(3*len(validation_samples)), nb_epoch=6)


# In[ ]:




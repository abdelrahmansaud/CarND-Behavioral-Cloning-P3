import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt

lines = []
with open('../data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        
car_images = []
steering_angles = []
for row in lines:
    path  = row[0]

    steering_center  = float(row[3])
    # create adjusted steering measurements for the side camera images
    correction = 0.15 # this is a parameter to tune
    steering_left = steering_center + correction
    steering_right = steering_center - correction
    
    
    img_center = np.asarray(cv2.imread(row[0]))
    img_left = np.asarray(cv2.imread(row[1]))
    img_right = np.asarray(cv2.imread(row[2]))
    
    # add images and angles to data set
    car_images.append(img_center)
    car_images.append(img_left)
    car_images.append(img_right)
    steering_angles.append(steering_center)
    steering_angles.append(steering_left)
    steering_angles.append(steering_right)
    
    
aug_images, aug_measurements = [], []
for image, measurement in zip(car_images, steering_angles):
    aug_images.append(image)
    aug_measurements.append(measurement)
    aug_images.append(cv2.flip(image, 1))
    aug_measurements.append(measurement * -1.0)
    
X_train = np.array(aug_images)
y_train = np.array(aug_measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Activation
from keras.layers.convolutional import Convolution2D, Cropping2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization

model  = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0) )))

model.add(Convolution2D(24,5,5, activation='relu', subsample=(2, 2)))
# model.add(MaxPooling2D())
model.add(Convolution2D(36,5,5, activation='relu', subsample=(2, 2)))
model.add(Convolution2D(48,5,5, activation='relu', subsample=(2, 2)))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))

model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
history_object = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=1)

model.save('model.h5')


### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
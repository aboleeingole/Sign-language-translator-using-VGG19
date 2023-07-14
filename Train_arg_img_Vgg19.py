# import  matplotlib.pyplot as plt
# import numpy as np
# import cv2
# import os
import PIL
# import cv2
# import numpy as np
# import tensorflow as tf
# from tensorflow import keras
# from keras import  layers
# from keras.models import Model
# from keras.applications import vgg19
# from pathlib import Path
# import PIL
# from sklearn.model_selection import train_test_split
# from keras.applications.vgg19 import VGG19
# from keras.layers import Flatten, Dense
# from keras.models import Model
#
# dataset_dir = 'augmented_images'
# dataset = Path(dataset_dir)
#
# # # to know the total images
# # num_images = len(list(dataset.glob('*/*.jpg')))
# # print(num_images)
#
# # Number images dictionary
#
# image_dict = {
#     '1' : list(dataset.glob('1/*.jpg')),
#     '2' : list(dataset.glob('2/*.jpg')),
#     '3': list(dataset.glob('3/*.jpg')),
# }
# image_labels = {
#     '1' : 1,
#     '2' : 2,
#     '3' : 3,
# }
# #
# # read the image from disk and covert to 3D numpy array
# img = cv2.imread(str(image_dict['1'][0]))
# # print(img)
# # print(img.shape) ## gives shape (x,y,rgb)
# # # # for  train make same foe all the images
# # # img_resize = cv2.resize(img,(100,100)).shape
# # # print(img_resize)
# #
# X , y = [], []
# for image_name, images in image_dict.items():
#     for image in images:
#         img = cv2.imread(str(image))
#         img_resize = cv2.resize(img, (224, 224))
#         q = X.append(img_resize)
#         b = y.append(image_labels[image_name])
# # # for my info
# # a = X[0]
# # print(a)
# #
# #
# #
# # convert python into num py array
# X = np.array(X)
# y = np.array(y)
# #
# # Train test spilt
# X_train ,X_test, y_train , y_test = train_test_split(X,y,random_state=0)
# #
# len(X_train)
# ## for knowing the length to train
# # a = len(X_train)
# # print(a)
# #
# #
# ## make range 0 and 1
# X_train_scale = X_train / 255
# X_test_scale = X_test / 255

# ## know thw values
# a = X_train_scale[0]
# print(a)
# num_classes = 3
# # Load the VGG19 model
# base_model = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
#
# # Freeze the layers of the pre-trained model
# for layer in base_model.layers:
#     layer.trainable = False
#
# # Add new trainable layers on top of the pre-trained model
# x = base_model.output
# x = Flatten()(x)
# x = Dense(4096, activation='relu')(x)
# x = Dense(4096, activation='relu')(x)
# predictions = Dense(num_classes, activation='softmax')(x)
#
# # Define the new model
# model = Model(inputs=base_model.input, outputs=predictions)
#
# # Compile the model
# model.compile(
#     optimizer='adam',
#     loss='categorical_crossentropy',
#     metrics=['accuracy']
# )
# # Train the model
# history = model.fit(
#     X_train_scale,
#     y_train,
#     epochs=10, # add number of epochs
#     steps_per_epoch=100,
#     verbose=1
# )
# # Save the model to a file
# model.save('vgg19_model.h5')
#
#

import numpy as np
import cv2
from tensorflow import keras
from keras.layers import Flatten, Dense
from keras.models import Model
from keras.applications.vgg19 import VGG19
from pathlib import Path
from sklearn.model_selection import train_test_split

model = keras.models.load_model('vgg19_model.h5')

dataset_dir = 'augmented_images'
dataset = Path(dataset_dir)

# Number images dictionary
image_dict = {
    '1': list(dataset.glob('1/*.jpg')),
    '2': list(dataset.glob('2/*.jpg')),
    '3': list(dataset.glob('3/*.jpg')),
    '4': list(dataset.glob('4/*.jpg')),
    '5': list(dataset.glob('5/*.jpg')),
    '6': list(dataset.glob('6/*.jpg')),
    '7': list(dataset.glob('7/*.jpg')),
    '8': list(dataset.glob('8/*.jpg')),
    '9': list(dataset.glob('9/*.jpg')),
    'A': list(dataset.glob('A/*.jpg')),
    'B': list(dataset.glob('B/*.jpg')),
    'C': list(dataset.glob('C/*.jpg')),
    'D': list(dataset.glob('D/*.jpg')),
    'E': list(dataset.glob('E/*.jpg')),
    'F': list(dataset.glob('F/*.jpg')),
    'G': list(dataset.glob('G/*.jpg')),
    'H': list(dataset.glob('H/*.jpg')),
    'I': list(dataset.glob('I/*.jpg')),
    'J': list(dataset.glob('J/*.jpg')),
    'K': list(dataset.glob('K/*.jpg')),
    'L': list(dataset.glob('L/*.jpg')),
    'M': list(dataset.glob('M/*.jpg')),
    'N': list(dataset.glob('N/*.jpg')),
    'O': list(dataset.glob('O/*.jpg')),
    'P': list(dataset.glob('P/*.jpg')),
    'Q': list(dataset.glob('Q/*.jpg')),
    'R': list(dataset.glob('R/*.jpg')),
    'S': list(dataset.glob('S/*.jpg')),
    'T': list(dataset.glob('T/*.jpg')),
    'U': list(dataset.glob('U/*.jpg')),
    'V': list(dataset.glob('V/*.jpg')),
    'W': list(dataset.glob('W/*.jpg')),
    'X': list(dataset.glob('X/*.jpg')),
    'Y': list(dataset.glob('Y/*.jpg')),
    'Z': list(dataset.glob('Z/*.jpg'))
}

image_labels = {
    '1': 0,
    '2': 1,
    '3': 2,
    '4': 3,
    '5': 4,
    '6': 5,
    '7': 6,
    '8': 7,
    '9': 8,
    'A': 9,
    'B': 10,
    'C': 11,
    'D': 12,
    'E': 13,
    'F': 14,
    'G': 15,
    'H': 16,
    'I': 17,
    'J': 18,
    'K': 19,
    'L': 20,
    'M': 21,
    'N': 22,
    'O': 23,
    'P': 24,
    'Q': 25,
    'R': 26,
    'S': 27,
    'T': 28,
    'U': 29,
    'V': 30,
    'W': 31,
    'X': 32,
    'Y': 33,
    'Z': 34
}

# Load the images and labels
X, y = [], []
for image_name, images in image_dict.items():
    for image in images:
        img = cv2.imread(str(image))
        img_resize = cv2.resize(img, (224, 224))
        X.append(img_resize)
        y.append(image_labels[image_name])

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

# Convert labels to one-hot encoded vectors
y = keras.utils.to_categorical(y, 35)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Scale the images
X_train_scale = X_train / 255
X_test_scale = X_test / 255

# Load the VGG19 model
base_model = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the layers of the pre-trained model
for layer in base_model.layers:
    layer.trainable = False

# Add new trainable layers on top of the pre-trained model
x = base_model.output
x = Flatten()(x)
x = Dense(4096, activation='relu')(x)
x = Dense(4096, activation='relu')(x)
predictions = Dense(35, activation='softmax')(x)

# Define the new model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    X_train_scale,
    y_train,
    epochs=5,
    verbose=1
)

# Save the model to a file
model.save('vgg19_model.h5')
# model = keras.models.load_model('vgg19_model.h5')

# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(X_test_scale, y_test)

print('Test loss:', test_loss)
print('Test accuracy:', test_accuracy)
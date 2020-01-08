import sys
import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.constraints import maxnorm
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Conv2D
from keras import callbacks
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import time
from keras.utils import np_utils
import LoadData


(X_train,y_train)=LoadData.data_set
(X_test,y_test)=LoadData.test_data_set

"""
Parameters
"""

#Define parameters
img_width,  img_height = 200,200
batch_size = 10
epochs = 150
nb_filters1 = 100
nb_filters2 = 100
nb_filters3 = 100
conv1_size = 3
conv2_size = 5
conv3_size = 7
pool_size = 2
classes_num = 3
# lr = 0.0001

model = Sequential()
model.add(Conv2D(nb_filters1, (conv1_size, conv1_size), activation="relu", border_mode ="same",input_shape=(img_width, img_height, 3),kernel_constraint=maxnorm(3)),)
# model.add(Activation(tf.nn.relu))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

model.add(Conv2D(nb_filters2, (conv2_size, conv2_size), activation="relu", border_mode ="same", kernel_constraint=maxnorm(3)))
# model.add(Activation(tf.nn.relu))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

model.add(Conv2D(nb_filters3, (conv3_size, conv3_size), activation="relu",border_mode ="same", kernel_constraint=maxnorm(3)))
# model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size), dim_ordering='th'))

model.add(Flatten())
model.add(Dense(512,kernel_constraint=maxnorm(3),activation="relu"))
model.add(Dropout(0.35))
model.add(Dense(2,activation="softmax"))


model.compile(loss='binary_crossentropy',optimizer=optimizers.adam(lr=0.001),metrics=['accuracy'])


log_dir = './tf-log/'
tb_cb = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0)
cbks = [tb_cb]

# normalize inputs from 0-255 to 0.0-1.0
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print("Number of classes ",y_train.shape)

model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,shuffle=True,callbacks=cbks)

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=1,batch_size=5)
print("Accuracy: %.2f%%" % (scores[1]*100))

#save the Model
model.save('model/model.h5')
model.save_weights('model/weights.h5')




"""


#Augmentation and data preperation

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('C:\Rahul\machine learning\Hachathon\AirprobeAssignment\data/train/',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('C:\Rahul\machine learning\Hachathon\AirprobeAssignment\data/test',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

model.fit_generator(training_set,
                         steps_per_epoch = 10,
                         epochs = 10,
                         validation_data = test_set,
                         validation_steps = 5)

#Predict the images
path = 'C:\Rahul\machine learning\Hachathon\AirprobeAssignment\data\model'
labels = os.listdir(path)
result = []
for images in labels:
    test_image = image.load_img(path+'/'+images, target_size = (64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result.append(model.predict(test_image))
    training_set.class_indices

print(result)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=1,batch_size=5)
print("Accuracy: %.2f%%" % (scores[1]*100))

#save the Model
model.save('models/model.h5')
model.save_weights('models/weights.h5')

target_dir = 'models/'
if not os.path.exists(target_dir):
  os.mkdir(target_dir)

#Calculate execution time
end = time.time()
dur = end-start

print("Execution Time:",dur,"seconds")
"""

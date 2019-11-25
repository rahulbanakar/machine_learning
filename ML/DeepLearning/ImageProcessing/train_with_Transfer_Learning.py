from keras import applications
import sys
import os
from keras.constraints import maxnorm
import tensorflow as tf
from keras import optimizers
from keras.models import Sequential,Model
from keras.layers import Dropout, Flatten, Dense, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras import callbacks
import time
import load_data
from keras.utils import np_utils

(X_train,y_train)=load_data.data_set
(X_test,y_test)=load_data.test_data_set

start = time.time()


"""
Parameters
"""
img_width,  img_height = 150, 150
batch_size = 10
epochs = 10
nb_filters1 = 50
nb_filters2 = 50
nb_filters3 = 50
conv1_size = 3
conv2_size = 3
conv3_size = 3
pool_size = 2
classes_num = 3
# lr = 0.0001

model = applications.ResNet50(include_top=False, weights='imagenet', input_shape=(img_width, img_height, 3))
for layer in model.layers[:]:
    layer.trainable = False

x = model.output
x = Flatten()(x)
x=Dense(512)(x)
x=Activation(tf.nn.relu)(x)
x=Dropout(0.4)(x)
predictions=Dense(3,activation="softmax")(x)
model_final = Model(input = model.input, output = predictions)

model_final.compile(loss='categorical_crossentropy',
              optimizer=optimizers.adam(lr=0.001),
              metrics=['accuracy'])


"""
Tensorboard log
"""
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

model_final.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size,shuffle=True,callbacks=cbks)

# Final evaluation of the model


target_dir = 'models/'
if not os.path.exists(target_dir):
  os.mkdir(target_dir)
model.save('models/model.h5')
model.save_weights('models/weights.h5')

scores = model.evaluate(X_test, y_test, verbose=1,batch_size=5)
print("Accuracy: %.2f%%" % (scores[1]*100))



#Calculate execution time
end = time.time()
dur = end-start

if dur<60:
    print("Execution Time:",dur,"seconds")
elif dur>60 and dur<3600:
    dur=dur/60
    print("Execution Time:",dur,"minutes")
else:
    dur=dur/(60*60)
    print("Execution Time:",dur,"hours")
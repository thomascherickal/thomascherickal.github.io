import numpy as np
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator

ROTATION_RANGE = 15
SHIFT_RANGE    = 0.1
BATCH_SIZE     = 100
EPOCHS         = 20

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32') / 255
Y_train = np_utils.to_categorical(y_train, 10)

X_test  = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32') / 255
Y_test  = np_utils.to_categorical(y_test, 10)

model = Sequential([
    Convolution2D(32, 3, 3, border_mode='valid', input_shape=(28, 28, 1)),
    Activation('relu'),
    Convolution2D(32, 5, 5, border_mode='valid'),
    Activation('relu'),
    Convolution2D(32, 7, 7, border_mode='valid'),
    Activation('relu'),
    MaxPooling2D(pool_size=(2,2)),
    Dropout(0.35),
    Flatten(),
    Dense(128),
    Activation('relu'),
    Dropout(0.5),
    Dense(10),
    Activation('softmax')
])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

dg = ImageDataGenerator(rotation_range=ROTATION_RANGE, width_shift_range=SHIFT_RANGE, height_shift_range=SHIFT_RANGE)
dg.fit(X_train)    
model.fit_generator(dg.flow(X_train, Y_train, batch_size=BATCH_SIZE), samples_per_epoch=len(X_train), 
                    nb_epoch=EPOCHS, validation_data=(X_test,Y_test))

score = model.evaluate(X_test, Y_test, verbose=0)
print('Accuracy:', score[1])

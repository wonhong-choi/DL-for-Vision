import keras
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.callbacks import ModelCheckpoint
import numpy as np
import matplotlib.pyplot as plt


def data_load():
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    return X_train, y_train, X_test, y_test

def data_preview(X_train):
    fig = plt.figure(figsize=(20,5))
    for i in range(36):
        ax=fig.add_subplot(3, 12, i+1, xticks=[], yticks=[])
        ax.imshow(np.squeeze(X_train[i]))

def normalize(X_train, X_test):
    X_train = X_train.astype('float32')/255
    X_test = X_test.astype('float32')/255
    return X_train, X_test

def one_hot(y_train, y_test):
    num_classes=len(np.unique(y_train))
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    return y_train, y_test

def split(X_train, y_train, valid_nums=5000):
    X_train, X_valid = X_train[valid_nums:], X_train[:valid_nums]
    y_train, y_valid = y_train[valid_nums:], y_train[:valid_nums]
    return X_train, X_valid, y_train, y_valid
    
def mini_AlexNet():
    cnn = Sequential()
    cnn.add(Conv2D(filters=16, kernel_size=2, padding='same',\
        activation='relu', input_shape = (32, 32, 3)))
    cnn.add(MaxPooling2D(pool_size=2))
    cnn.add(Conv2D(filters=32, kernel_size=2, padding='same',\
        activation='relu', input_shape = (32, 32, 3)))
    cnn.add(MaxPooling2D(pool_size=2))
    cnn.add(Conv2D(filters=64, kernel_size=2, padding='same',\
        activation='relu', input_shape = (32, 32, 3)))
    cnn.add(MaxPooling2D(pool_size=2))
    
    cnn.add(Dropout(rate=0.3))

    cnn.add(Flatten())

    cnn.add(Dense(500, activation='relu'))
    cnn.add(Dropout(rate=0.4))

    cnn.add(Dense(10, activation='softmax'))
    return cnn


if __name__ == "__main__":
    X_train, y_train, X_test, y_test=data_load()

    X_train, X_test = normalize(X_train, X_test)

    y_train, y_test = one_hot(y_train, y_test)

    X_train, X_valid, y_train, y_valid = split(X_train, y_train)
    
    model=mini_AlexNet()
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop',\
        metrics=['accuracy'])

    checkpointer = ModelCheckpoint(filepath='model.weights.best.hdf5', verbose=1, save_best_only=True)

    hist= model.fit(X_train, y_train, batch_size=32, epochs=100,\
        validation_data=(X_valid, y_valid), callbacks=[checkpointer],\
            verbose=2, shuffle=True)
    
    
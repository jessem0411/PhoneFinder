import sys
import os

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import BatchNormalization
from keras.layers.core import Activation, Flatten, Dropout, Dense

def train_phone_finder(path):
    os.chdir(path)
    cwd = os.getcwd()
    data = []

    for file in os.listdir(cwd):
        if file.endswith(".txt"):
            with open("labels.txt") as file:
                for row in file:
                    data_row = [l.strip() for l in row.split(' ')]
                    data.append(data_row)
    pix = []
    loc = []
    for name in data:
        image = cv2.imread(name[0])
        resized_img = cv2.resize(image, (96,96))
        pix.append(resized_img.tolist())
        x = float(name[1])
        y = float(name[2])
        loc.append([x, y])
    pix = np.asarray(pix)
    loc = np.asarray(loc)
    pix = np.interp(pix, (pix.min(), pix.max()), (0, 1))

    (x_train, x_test, y_train, y_test) = train_test_split(pix, loc,
                                                          test_size=0.08, random_state=13)

    input_shape = (pix.shape[-2], pix.shape[-3], pix.shape[-1])
    model = Sequential()
    model.add(Conv2D(64, (3, 3), activation='relu', padding="same", input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.35))

    model.add(Conv2D(128, (3, 3), activation='relu', padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.35))

    model.add(Conv2D(128, (3, 3), activation='relu', padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.35))

    model.add(Conv2D(128, (3, 3), activation='relu', padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.35))

    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation("sigmoid"))

    model.add(Dense(128))
    model.add(Activation("sigmoid"))

    model.add(Dense(loc.shape[-1]))
    model.compile(loss="mse", optimizer='adam', metrics=["accuracy"])
    model.fit(x_train,y_train,validation_data=(x_test,y_test), epochs=50, batch_size=8)
    model.save('train_phone_finder_results.h5')
    print("Training complete.")

def main():
    train_phone_finder(sys.argv[1])

if __name__ == "__main__":
    main()
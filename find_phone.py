import os
import sys

import cv2
import numpy as np

from keras.models import load_model

def find_phone(path):
    filename = list(os.path.split(path))[1]
    pathname = list(os.path.split(path))[0]
    os.chdir(pathname)

    pix = []
    image = cv2.imread(filename)
    resized_image = cv2.resize(image,(96,96))
    pix.append(resized_image.tolist())
    pix = np.asarray(pix)
    pix = np.interp(pix, (pix.min(), pix.max()), (0, 1))
    os.chdir('..')
    model = load_model('find_phone/train_phone_finder_results.h5')
    result = model.predict(pix)
    print("{:.4f} {:.4f}".format(result[0][0],result[0][1]))

def main():
    find_phone(sys.argv[1])

if __name__ == "__main__":
    main()
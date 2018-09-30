import cv2 as cv
import numpy as np


def stackChannels(X):
    h = X.shape[0] // 3
    X_ = [X[:h], X[h:2 * h], X[2 * h:3 * h]]
    return np.dstack(X_)


def normalize(X):
    Xmax = np.iinfo(X.dtype).max
    return X.astype(np.double) / Xmax


def load(path):
    format = cv.CV_8UC1 if 'jpg' in path else cv.CV_16UC1
    return cv.imread(path, format)


def save(X, path):
    cv.imwrite(path, X)


def show(X, dismiss='any', **kwargs):
    title = kwargs.get('title', None)
    persist = kwargs.get('persist', False)
    key = None

    cv.namedWindow(title, cv.WINDOW_KEEPRATIO)
    cv.imshow(title, X)
    key = cv.waitKey()

    while key != ord(dismiss) and dismiss != 'any':
        key = cv.waitKey()

    if not persist:
        cv.destroyWindow(title)

    return key

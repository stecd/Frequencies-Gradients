import numpy as np
from scipy import sparse
import cv2 as cv
import matplotlib.pyplot as plt
from utils import Profiler




def init():

    im = plt.imread('inputs/toy_problem.png')
    im2var = np.arange(im.shape[0] * im.shape[1]).reshape(*im.shape[0:2])

    numPx = im.shape[0] * im.shape[1]
    numEq = 2 * numPx +1

    A = sparse.csr_matrix((numEq, numPx)).tolil()
    b = np.zeros((numEq))

    i = np.repeat(np.arange(im.shape[0]), im.shape[1] - 1)
    j = np.tile(np.arange(im.shape[1] -1), im.shape[0])
    e = np.arange(im.shape[0] * (im.shape[1] - 1))

    A[e, im2var[i, j + 1]] = 1
    A[e, im2var[i, j]] = -1
    b[e] = im[i, j + 1] - im[i, j]

    A[-1, im2var[0, 0]] = 1
    b[-1] = im[0, 0]

    V = sparse.linalg.lsqr(A.tocsr(), b)
    V = np.array(V[0]).reshape(*im.shape)

    plt.imshow(V, cmap='Greys_r', aspect='equal')
    plt.show()




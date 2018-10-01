import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

N = 6

class StackMode:
    Gauss = 0
    Laplace = 1

def stack(X, levels, mode: StackMode, sigma):
    kSize = (4 * sigma + 1, 4 * sigma + 1)

    g = [X]

    for i in range(levels-1):
        X_ = cv.GaussianBlur(g[i], kSize, sigmaX=sigma)
        g.append(X_)

    if mode is StackMode.Laplace:
        l = []

        for i in range(levels - 1):
            l.append(g[i] - g[i + 1])

        l.append(g[-1])

        return np.dstack(l)
    else:
        return np.dstack(g)

def stackColor(X, levels, mode: StackMode, sigma):
    kSize = tuple([4 * sigma + 1]) * 2
    l = [X]

    if mode == StackMode.Gauss:
        for _ in range(levels-1):
            r = cv.GaussianBlur(l[-1][:, :, 0], kSize, 0)
            g = cv.GaussianBlur(l[-1][:, :, 1], kSize, 0)
            b = cv.GaussianBlur(l[-1][:, :, 2], kSize, 0)
            l.append(np.dstack([r, g, b]))
    else:
        for _ in range(levels - 1):
            r = cv.GaussianBlur(l[-1][:, :, 0], kSize, 0)
            g = cv.GaussianBlur(l[-1][:, :, 1], kSize, 0)
            b = cv.GaussianBlur(l[-1][:, :, 2], kSize, 0)
            l.append(np.clip(l[-1] - np.dstack([r, g, b]), 0, 1))

        r = cv.GaussianBlur(l[-1][:, :, 0], kSize, 0)
        g = cv.GaussianBlur(l[-1][:, :, 1], kSize, 0)
        b = cv.GaussianBlur(l[-1][:, :, 2], kSize, 0)
        l.append([r, g, b])

        l.pop(0)

    return l

def greyScale():
    img = plt.imread('inputs/lincoln.jpg')
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY) / 255

    G = stack(img, levels=N, mode=StackMode.Gauss, sigma=5)
    L = stack(img, levels=N, mode=StackMode.Laplace, sigma=5)

    fig = plt.figure(figsize=(25, 7))
    fig.subplots_adjust(hspace=0.1, wspace=0.1)

    for i in range(1, N+1):
        ax = fig.add_subplot(2, N, i)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(G[:, :, i - 1], cmap='Greys_r', aspect='auto')
        ax.text(0, -20, r'$Gaussian_{}$'.format(i-1))

        ax = fig.add_subplot(2, N, N + i)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(L[:, :, i - 1], cmap='Greys_r', aspect='auto')
        ax.text(0, -20, r'$Laplace_{}$'.format(i - 1))

    plt.show()

def init():
    greyScale()


init()
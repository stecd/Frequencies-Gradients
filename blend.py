import matplotlib.pyplot as plt
import numpy as np
import utils.ops as ops
import stacks
import cv2 as cv




def blend(A, B, levels, l, g, mask):

    LA = stacks.stack(A, levels=levels, mode=stacks.StackMode.Laplace, kernel=l)
    LB = stacks.stack(B, levels=levels, mode=stacks.StackMode.Laplace, kernel=l)
    GM = stacks.stack(mask, levels=levels, mode=stacks.StackMode.Gauss, kernel=g)

    idx = tuple([slice(None)] * len(LA.shape))
    LS = np.zeros_like(LA)
    LS[idx] = GM[idx] * LA[idx] + (1 - GM[idx]) * LB[idx]

    S = np.clip(np.sum(LS, axis=len(LS.shape) - 1), 0, 1)
    return S, LA, LB, LS, GM

def getMask(size):
    mask = np.zeros_like(size)
    mask[:, :mask.shape[0] // 2] = 1
    return mask

def showLaplace(LA, LB, LS, GM, levels):

    # LA[:, :, :-1] = LA[:,:, :-1] + 0.1
    # LB[:, :, :-1] = LB[:, :, :-1] + 0.1
    # LS[:, :, :-1] = LB[:, :, :-1] + 0.1

    fig = plt.figure(figsize=(5, 5))
    fig.subplots_adjust(hspace=0.4, wspace=0)

    col = 4

    for i in range(levels):
        ax = fig.add_subplot(levels, col, 1 + col*i)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(LA[:,:, i], aspect='equal', cmap='Greys_r')
        ax.text(0, -20, r'$L_{A%i}$' % i)

        ax = fig.add_subplot(levels, col, 2 + col*i)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(LS[:,:, i], aspect='equal', cmap='Greys_r')
        ax.text(0, -20, r'$L_{S%i}$' % i)

        ax = fig.add_subplot(levels, col, 3 + col* i)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(LB[:,:,i], aspect='equal', cmap='Greys_r')
        ax.text(0, -20, r'$L_{B%i}$' % i)

        ax = fig.add_subplot(levels, col, 4 + col* i)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(GM[:, :, i], aspect='equal', cmap='Greys_r')
        ax.text(0, -20, r'$G_{M%i}$' % i)


def init():
    N = 4
    # A = ops.normalize(plt.imread('inputs/tatoo.jpg'))
    # B = ops.normalize(plt.imread('inputs/ste_tattoo.jpg'))

    A = ops.normalize(cv.cvtColor(plt.imread('inputs/tatoo.jpg'), cv.COLOR_BGR2GRAY))
    B = ops.normalize(cv.cvtColor(plt.imread('inputs/ste_tattoo.jpg'), cv.COLOR_BGR2GRAY))

    # mask = getMask(A)
    mask = ops.normalize(cv.cvtColor(plt.imread('inputs/mask_tattoo.jpg'), cv.COLOR_BGR2GRAY))

    S, LA, LB, LS, GM = blend(A, B, levels=N, l=21, g=61, mask=mask)

    fig = plt.figure(figsize=(5, 5))
    fig.subplots_adjust(hspace=0.4, wspace=0)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(S, aspect='equal', cmap='Greys_r')

    # showLaplace(LA, LB, LS, GM, N)

    plt.show()




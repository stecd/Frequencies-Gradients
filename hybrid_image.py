import matplotlib.pyplot as plt
from align_image_code import align_images
import numpy as np
import cv2 as cv
import utils.ops as ops


def align(im1, im2):
    im1_aligned, im2_aligned = align_images(im1, im2)
    # plt.imshow(im1_aligned * 0.5 + im2_aligned * 0.5)
    # plt.show()
    return im1_aligned, im2_aligned


def plotShow(data):
    plt.imshow(data, cmap="Greys_r")
    plt.show()


def plotFFTShowAll(im1_gray, im2_gray, hybrid, val1, val2):
    #compute and display all the 2D Fourier transforms
    lowPass = cv.GaussianBlur(im1_gray, (val1, val2), sigmaX=0)
    highPass = im2_gray - cv.GaussianBlur(im2_gray, (val1, val2), sigmaX=0)
    plotShow(fft(im1_gray))
    plotShow(fft(im2_gray))
    plotShow(fft(lowPass))
    plotShow(fft(highPass))
    plotShow(fft(hybrid))


def hybrid_image(highPass, lowPass, val1, val2):
    # k1, k2 = 4 * s1 + 1, 4 * s2 + 1
    # k1, k2 = int((20*s1-7)/3), int((20*s2-7)/3)
    k1, k2 = val1, val2
    lowPass_im = cv.GaussianBlur(lowPass, (k1, k1), sigmaX=0)
    highPass_im = highPass - cv.GaussianBlur(highPass, (k2, k2), sigmaX=0)

    k = 0.5
    return (lowPass_im * k) + (highPass_im * (1 - k))


def fft(data):
    return np.log(np.abs(np.fft.fftshift(np.fft.fft2(data))))


def init():
    im1 = plt.imread('inputs/monkey.jpg')
    im2 = plt.imread('inputs/monkey_ste.JPG')
    k1 = 29 # MUST BE ODD
    k2 = 61 # MUST BE ODD
    # im1, im2 = align(im1, im2)

    # # ~~~~~~~~~ GRAYSCALE
    # im1_gray = cv.cvtColor((255*im1).astype(np.uint8), cv.COLOR_BGR2GRAY)/255
    # im2_gray = cv.cvtColor((255*im2).astype(np.uint8), cv.COLOR_BGR2GRAY)/255
    # hybrid = hybrid_image(highPass=im1_gray, lowPass=im2_gray, val1=k1, val2=k2)

    # ~~~~~~~~~ COLOR
    # BGR = [
    #   hybrid_image(im1[:, :, 0], im2[:, :, 0], k1, k2),
    #   hybrid_image(im1[:, :, 1], im2[:, :, 1], k1, k2),
    #   hybrid_image(im1[:, :, 2], im2[:, :, 2], k1, k2)
    # ]

    # ~~~~~~~~~ COLOR+bw - only lower freq layer color.
    # im1_aligned, im2_aligned = align(im1, im2)
    # im1_gray = cv.cvtColor((255*im1_aligned).astype(np.uint8), cv.COLOR_BGR2GRAY)/255
    # BGR = [
    #   hybrid_image(im1_gray, im2_aligned[:, :, 0], k1, k2),
    #   hybrid_image(im1_gray, im2_aligned[:, :, 1], k1, k2),
    #   hybrid_image(im1_gray, im2_aligned[:, :, 2], k1, k2)
    # ]
    # hybrid = np.dstack(reversed(BGR))


    # ~~~~~~~~~ COLOR+bw - only high freq layer color.
    im1_aligned, im2_aligned = align(im1, im2)
    im2_gray = cv.cvtColor((255*im2_aligned).astype(np.uint8), cv.COLOR_BGR2GRAY)/255
    BGR = [
      hybrid_image(im1_aligned[:, :, 0], im2_gray, k1, k2),
      hybrid_image(im1_aligned[:, :, 1], im2_gray, k1, k2),
      hybrid_image(im1_aligned[:, :, 2], im2_gray, k1, k2)
    ]
    hybrid = np.dstack(reversed(BGR))

    # plotFFTShowAll(im1_gray, im2_gray, hybrid, k1, k2)
    ops.show(hybrid, title='hybrid')

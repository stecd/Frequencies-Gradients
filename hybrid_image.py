import matplotlib.pyplot as plt
from align_image_code import align_images
import numpy as np
import cv2 as cv
import utils.ops as ops


def align():
    im1_aligned, im2_aligned = align_images(im1, im2)
    # plt.imshow(im1_aligned * 0.5 + im2_aligned * 0.5)
    # plt.show()
    return im1_aligned, im2_aligned


def plotShow(input):
    plt.imshow(input, cmap="Greys_r")
    plt.show()


def plotFFTShowAll():
    #compute and display all the 2D Fourier transforms
    lowPass = cv.GaussianBlur(im1_gray, (35, 35), sigmaX=sigma1)
    highPass = im2_gray - cv.GaussianBlur(im2_gray, (35, 35), sigmaX=sigma1)
    plotShow(fft(im1_gray))
    plotShow(fft(im2_gray))
    plotShow(fft(lowPass))
    plotShow(fft(highPass))
    plotShow(fft(hybrid))


def hybrid_image(im1, im2, s1, s2):
    # k1, k2 = 4 * s1 + 1, 4 * s2 + 1
    k1, k2 = int((20*s1-7)/3), int((20*s2-7)/3)
    lowPass_im1 = cv.GaussianBlur(im1, (k1, k1), sigmaX=s1)
    highPass_im2 = im2 - cv.GaussianBlur(im2, (k2, k2), sigmaX=s2)

    k = 0.5
    return (lowPass_im1 * k) + (highPass_im2 * (1 - k))


def fft(input):
    return np.log(np.abs(np.fft.fftshift(np.fft.fft2(input))))


########################################## Part 1.2: Hybrid Images
im1 = plt.imread('inputs/ste_sad.jpg') / 255
im2 = plt.imread('inputs/ste_happy.jpg') / 255
sigma1 = 15
sigma2 = 30

# # ~~~~~~~~~ GRAYSCALE
# im1_aligned, im2_aligned = align()
# im1_gray = cv.cvtColor((255*im1_aligned).astype(np.uint8), cv.COLOR_BGR2GRAY)/255
# im2_gray = cv.cvtColor((255*im2_aligned).astype(np.uint8), cv.COLOR_BGR2GRAY)/255
# hybrid = hybrid_image(im1_gray, im2_gray, sigma1, sigma2)

# ~~~~~~~~~ COLOR
im1, im2 = align()

# BGR = [
#   hybrid_image(im1[:, :, 0], im2[:, :, 0], sigma1, sigma2),
#   hybrid_image(im1[:, :, 1], im2[:, :, 1], sigma1, sigma2),
#   hybrid_image(im1[:, :, 2], im2[:, :, 2], sigma1, sigma2)
# ]

im1_gray = cv.cvtColor((255*im1).astype(np.uint8), cv.COLOR_BGR2GRAY)/255
BGR = [
  hybrid_image(im1_gray, im2[:, :, 0], sigma1, sigma2),
  hybrid_image(im1_gray, im2[:, :, 1], sigma1, sigma2),
  hybrid_image(im1_gray, im2[:, :, 2], sigma1, sigma2)
]

hybrid = np.dstack(reversed(BGR))


ops.show(hybrid, title='hybrid')
# plotFFTShowAll()


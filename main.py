import numpy as np
import cv2 as cv
import utils.ops as ops


def unsharp(img, bimg, alpha=0.9 ):
  return img + alpha * (img - bimg)

def part1_1():
  img = ops.load('inputs/alcatraz2.jpg')
  img = ops.normalize(img)

  blurred_img = cv.GaussianBlur(img, (5, 5), 0)
  sharp_img = unsharp(img, blurred_img)

  # cv.imshow('img', img)
  # cv.imshow('blurred_img', blurred_img)
  # cv.imshow('sharp_img', sharp_img)
  # cv.waitKey()

  ops.save((sharp_img*255).astype(np.uint64), 'outputs/alcatraz_sharp_p9.jpg')
  ops.save((img*255).astype(np.uint64), 'outputs/alcatraz.jpg')


def init():
  part1_1()


init()

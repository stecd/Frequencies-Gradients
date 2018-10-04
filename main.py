import numpy as np
import cv2 as cv
import utils.ops as ops
import hybrid_image as hybrid
import stacks
import blend
import gradient
import poisson


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

def part1_2():
    hybrid.init()

def part1_3():
    stacks.init()

def part1_4():
    blend.init()

def part2_1():
    gradient.init()

def part2_2():
    poisson.init()

def init():
  # part1_1()
  # part1_2()
  # part1_3()
  # part1_4()
  # part2_1()
  part2_2()

init()

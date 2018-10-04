from scipy import sparse as spr
from scipy.sparse import linalg as spr_linalg
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from utils import ops
from utils import Profiler


def makeSubMask(M, contour, pad):
    x, y, w, h = cv.boundingRect(contour)
    C = np.pad(
        M[y:y + h, x:x + w],
        ((pad if y > 0 else 0, pad if y + h > 0 else 0),
         (pad if x > 0 else 0, pad if x + w > 0 else 0)),
        'constant'
    )

    return (y, x), C


def extractMasks(M, pad, includeEnclosing=False):
    M_ = (255 * M).astype(np.uint8)
    _, contours, _ = cv.findContours(M_, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    bounds = [makeSubMask(M, c, pad) for c in contours]

    if includeEnclosing:
        allContours = np.concatenate(contours)
        bounds.append(makeSubMask(M, allContours, pad))

    return bounds


def gBlendMulti(S, T, M, mode, yxT=None):
    masks = extractMasks(M, 1, includeEnclosing=True)[-1:]
    T_ = T.copy()

    for ((yS, xS), M) in masks:
        R = gBlend(S, T, M, mode, yxS=(yS, xS), yxT=yxT)
        yT, xT = yxT or (yS, xS)
        hT, wT = R.shape[0:2]
        T_[yT:yT + hT, xT:xT + wT] = R

    return T_


def gBlend(S, T, M, mode, **kwargs):
    yS, xS = kwargs.get('yxS', (0, 0))
    yT, xT = kwargs.get('yxT', None) or yS, xS
    hS, wS = M.shape[0:2]
    hS, wS = min(hS, T.shape[0]), min(wS, T.shape[1])

    iM_, jM_ = np.where(M == 0)
    m_Num = len(iM_)
    pxNum = hS * wS
    eqNum = m_Num + hS * (wS - 1) + (hS - 1) * wS

    I2V = np.arange(pxNum).reshape(hS, wS)
    A = spr.csr_matrix((eqNum, pxNum)).tolil()
    b = np.zeros(eqNum)

    i = np.repeat(np.arange(hS), wS - 1)
    j = np.tile(np.arange(wS - 1), hS)
    e = np.arange(hS * (wS - 1))

    A[e, I2V[i, j + 1]] = 1
    A[e, I2V[i, j]] = -1
    dt = T[i + yT, j + xT + 1] - T[i + yT, j + xT]
    ds = S[i + yS, j + xS + 1] - S[i + yS, j + xS]

    b[e] = np.where(
        (M[i, j] == 0) & (M[i, j + 1] == 0),
        dt, np.maximum(ds, ds if mode is 1 else dt)
    )

    i = np.repeat(np.arange(hS - 1), wS)
    j = np.tile(np.arange(wS), hS - 1)
    e = np.arange((hS - 1) * wS) + e[-1] + 1

    A[e, I2V[i + 1, j]] = 1
    A[e, I2V[i, j]] = -1
    dt = T[i + yT + 1, j + xT] - T[i + yT, j + xT],
    ds = S[i + yS + 1, j + xS] - S[i + yS, j + xS]

    b[e] = np.where(
        (M[i, j] == 0) & (M[i + 1, j] == 0),
        dt, np.maximum(ds, ds if mode is 1 else dt)

    )

    e = np.arange(m_Num) + e[-1] + 1
    A[e, I2V[iM_, jM_]] = 1
    b[e] = T[iM_ + yT, jM_ + xT]

    T_ = spr_linalg.lsqr(A.tocsr(), b)
    T_ = np.array(T_[0]).reshape(hS, wS)

    return T_

def init():

    M = plt.imread('inputs/blends/pidgey_0000_mask.png').astype(np.float)
    S = ops.normalize(plt.imread('inputs/blends/pidgey_0001_bird.jpg'))
    T = ops.normalize(plt.imread('inputs/blends/pidgey_0002_flower.jpg'))

    # X = S * M + (1-M) * T

    B = gBlend(S[:, :, 0], T[:, :, 0], M[:, :, 0], mode=1)
    G = gBlend(S[:, :, 1], T[:, :, 1], M[:, :, 1], mode=1)
    R = gBlend(S[:, :, 2], T[:, :, 2], M[:, :, 2], mode=1)

    # B = X[:,:,0]
    # G = X[:,:,1]
    # R = X[:,:,2]

    X = np.dstack([R, G, B])
    cv.imshow('', X)
    cv.waitKey()


# Visual Computing: Wavelets for Computer Graphics
# team01

import numpy as np
from PIL import Image
import math

#TODO: needs return? -->YES
#TODO: same for non-standard


def kd_test_decomp_recon_on_1d_array():
    c=np.array([9.,7.,2.,6.])
    print(c)
    print("decomp")
    c=_normalized_decomposition(c, False)
    print(c)
    print("recon")
    c=_normalized_reconstruction(c, False)
    print(c)
    return
def kd_test_decomp_recon_on_image():
    # load lenna as 2d grayscale array
    image_values = np.array(Image.open('img/Lenna.png').convert('L'))
    image_values=image_values.astype(np.double)

    # std decom
    print("decomp")
    image_values=normalized_decomposition_2d(image_values)
    # save img
    #im = Image.fromarray(image_values)
    #im.save("img/Lenna_DECOMP.png")

    # std recon
    print("recon")
    image_values=normalized_reconstruction_2d(image_values)
    # save img
    image_values=image_values.astype(np.uintc)
    im = Image.fromarray(image_values)
    im.save("img/Lenna_RECON.png")

    return


def normalized_decomposition_2d(image_values, normalized=True):
    # Haar decomposition of a 2D array inplace
    print(image_values.shape)
    for i in range(0, image_values.shape[0]):
        image_values[i] = _normalized_decomposition(image_values[i], normalized)
    for i in range(0, image_values.shape[1]):
        image_values[:, i] = _normalized_decomposition(image_values[:, i], normalized)
    return image_values

    for i, row in enumerate(image_values):
        image_values[i] = _normalized_decomposition(row, normalized)

    for i, col in enumerate(image_values.T):
        image_values[:, i] = _normalized_decomposition(col, normalized)

    return image_values


def normalized_reconstruction_2d(image_values, normalized=True):
    # Haar reconstruction of a 2D array inplace
    print(image_values.shape)

    for i in range(0, image_values.shape[1]):
        image_values[:, i] = _normalized_reconstruction(image_values[:, i], normalized)
    for i in range(0, image_values.shape[0]):
        image_values[i] = _normalized_reconstruction(image_values[i], normalized)
    return image_values

    for i, col in enumerate(image_values.T):
        image_values[:, i] = _normalized_reconstruction(col, normalized)

    for i, row in enumerate(image_values):
        image_values[i] = _normalized_reconstruction(row, normalized)

    return image_values

def _normalized_decomposition(coefficients, normalized=True):
    # Haar decomposition of array inplace

    #normalize
    if normalized:
        sqrt2=math.sqrt(2)
        #sqrt2=1
        np.divide(coefficients,sqrt2)

    #start with whole array
    until=coefficients.size

    while until>=2:
        #print(coefficients)
        coefficients=_normalized_decomposition_step(coefficients, until, normalized)
        until/=2

    return coefficients

def _normalized_decomposition_step(coefficients, until, normalized=True):
    # One step of Haar decomposition of array inplace

    # apply changes on copy first
    copy=np.copy(coefficients)

    # if normalized devide by sqrt(2) else take avg
    if normalized:
        div_sqrt_2 = 1.0/math.sqrt(2)
    else:
        div_sqrt_2=1.0/2.0

    detail_i=int((until)/2)
    for i in range(0,detail_i):
        # make sure, values can be negative (default uint)
        val1=int(coefficients[2*i])
        val2=int(coefficients[2*i+1])
        avg=(val1+val2)*div_sqrt_2
        detail=(val1-val2)*div_sqrt_2
        copy[i]=avg
        copy[detail_i+i]=detail

    # apply changes to original array
    coefficients=copy
    return coefficients

def _normalized_reconstruction(coefficients, normalized=True):
    # Haar reconstruction of array inplace

    # start with 2 and go up to whole size
    until = 2
    while until <= coefficients.size:
        #print(coefficients)
        coefficients = _normalized_reconstruction_step(coefficients, until, normalized)
        until *= 2

    # undo normalize
    if normalized:
        sqrt2 = math.sqrt(2)
        np.multiply(coefficients, sqrt2)

    return coefficients

def _normalized_reconstruction_step(coefficients, until, normalized=True):
    # One step of Haar reconstruction of array inplace

    # apply changes on copy first
    copy = np.copy(coefficients)

    if normalized:
        div_sqrt_2 = 1.0 / math.sqrt(2)
    else:
        div_sqrt_2 = 1

    detail_index = int((until) / 2)
    for i in range(0, detail_index):
        # make sure, values can be negative (default uint)
        avg = int(coefficients[i])
        detail = int(coefficients[detail_index+i])
        val1 = (avg+detail)*div_sqrt_2
        val2 = (avg-detail)*div_sqrt_2
        copy[2*i] = val1
        copy[2*i+1] = val2

    # apply changes to original array
    coefficients = copy
    return coefficients





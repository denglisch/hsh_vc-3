# Visual Computing: Wavelets for Computer Graphics
# team01

import numpy as np
from PIL import Image
import math

#TODO: needs return?
#TODO: same for non-standard


def kd_test():
    c=np.array([9,7,2,6])
    print(c)
    print("decom")
    c=_normalized_decomposition(c)
    print(c)
    print("recon")
    c=_normalized_reconstruction(c)
    print(c)
    return


def normalized_decomposition_2d(image_values:np.array):
    # Haar decomposition of a 2D array inplace
    shape=image_values.shape()
    #(row (height), column (width), color (channel))

    width=shape[1]
    height=shape[0]

    for row in image_values:
        _normalized_decomposition(row)

    return image_values

def normalized_reconstruction_2d(image_values):
    # Haar reconstruction of a 2D array inplace
    return

def _normalized_decomposition(coefficients:np.array):
    # Haar decomposition of array inplace

    #normalize
    l_2j=coefficients.size
    sqrt2=math.sqrt(2)
    sqrt2=1
    np.divide(coefficients,sqrt2)

    until=l_2j
    while until>=2:
        #print(coefficients)
        coefficients=_normalized_decomposition_step(coefficients, until)
        until/=2

    return coefficients

def _normalized_decomposition_step(coefficients, until):
    # One step of Haar decomposition of array inplace
    copy=np.copy(coefficients)
    div_sqrt_2 = 1.0/math.sqrt(2)
    div_sqrt_2=1.0/2.0
    top=int((until)/2)
    for i in range(0,top):
        val1=coefficients[2*i]
        val2=coefficients[2*i+1]
        avg=(val1+val2)*div_sqrt_2
        detail=(val1-val2)*div_sqrt_2
        copy[i]=avg
        copy[top+i]=detail

    coefficients=copy
    return coefficients

def _normalized_reconstruction(coefficients):
    # Haar reconstruction of array inplace

    l_2j = coefficients.size

    until = 2
    while until <= l_2j:
        print(coefficients)
        coefficients = _normalized_reconstruction_step(coefficients, until)
        until *= 2

    # undo normalize
    sqrt2 = math.sqrt(2)
    sqrt2 = 1
    np.multiply(coefficients, sqrt2)

    return coefficients

def _normalized_reconstruction_step(coefficients, until):
    # One step of Haar reconstruction of array inplace
    copy = np.copy(coefficients)
    div_sqrt_2 = 1.0 / math.sqrt(2)
    div_sqrt_2 = 1
    top = int((until) / 2)
    for i in range(0, top):
        avg = coefficients[i]
        detail = coefficients[top+i]
        val1 = (avg+detail)*div_sqrt_2
        val2 = (avg-detail)*div_sqrt_2
        copy[2*i] = val1
        copy[2*i+1] = val2

    coefficients = copy
    return coefficients





# Visual Computing: Wavelets for Computer Graphics
# team01

import numpy as np
from PIL import Image
import math

#TODO: needs return? -->YES
#TODO: same for non-standard

mini=0
maxi=0
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

    #print(image_values[0][0].dtype)
    # make sure, values can be negative (default uint)
    image_values=image_values.astype(np.float64)
    #print(image_values[0])

    # std decomp
    print("decomp")
    global mini
    mini=0
    global maxi
    maxi=0
    image_values=decomposition_2d(image_values, normalized=True, standard=False)
    #print(image_values[0])
    # save img

    #image_values=np.add(image_values, 100.0)

    copy=np.copy(image_values)
    copy=np.multiply(copy, _read_min_dim(image_values))
    #copy=np.add(copy, 128.0)
    #print(copy[10])
    copy=copy.astype(np.uint8)
    im = Image.fromarray(copy)
    im.save("img/Lenna_DECOMP.png")
    print("min: {} max: {}".format(mini,maxi))

    #image_values=np.subtract(image_values, 100.0)

    #image_values=image_values.astype(np.double)

    # std recon
    print("recon")
    image_values=reconstruction_2d(image_values, normalized=True, standard=False)
    #print(image_values[0])
    # save img
    image_values=image_values.astype(np.uint8)
    im = Image.fromarray(image_values)
    im.save("img/Lenna_RECON.png")

    return

def kd_test_compression_on_image():
    # load lenna as 2d grayscale array
    image_values = np.array(Image.open('img/Lenna.png').convert('L'))

    # make sure, values can be negative (default uint)
    image_values=image_values.astype(np.float64)

    global mini
    mini=0
    global maxi
    maxi=0

    print("compression")
    image_values=compression(image_values)

    # save img
    image_values=image_values.astype(np.uint8)
    im = Image.fromarray(image_values)
    im.save("img/Lenna_COMP.png")

    return

def compression(image_values):
    # return compressed image_values

    # normalized 2d decomp.
    image_values=decomposition_2d(image_values, normalized=True, standard=True)
    # one array of coefficients
    coefficients=image_values.flatten()
    # sort by decreasing magnitude
    # commented since it can become very slow
    # coefficients[::-1].sort()

    # either by max error
    thres_min=np.abs(coefficients).min() # abs(coefficients[len(coefficients)-1])
    thres_max=np.abs(coefficients).max() # abs(coefficients[0])
    threshold = (thres_min + thres_max) / 2.
    epsilon_squared=20000000.0
    # find threashold
    found_thres=False
    while not found_thres:
        threshold=(thres_min+thres_max)/2.
        sum_squared=0
        for i in range(0, coefficients.size):
            if abs(coefficients[i])<threshold:
                sum_squared+=math.pow(coefficients[i],2)
        if sum_squared<epsilon_squared:
            thres_min=threshold
        else:
            thres_max=threshold
        #until thres_min=thres_max
        found_thres=(thres_max-thres_min)<0.1


    # or max coefficients
    # returning threshold also?

    # truncate small values in image_values
    count=0
    for (row, col), value in np.ndenumerate(image_values):
        if abs(value)<threshold:
            image_values[row, col]=0
            count+=1
    print("truncated {} values ({} %)".format(count, (count/coefficients.size*100)))

    # recon image
    image_values=reconstruction_2d(image_values, normalized=True, standard=True)

    # after return
    # save image
    return image_values

def decomposition_2d(image_values, normalized=True, standard=True):
    # Haar decomposition of a 2D array inplace
    #print(image_values.shape)
    if standard:
        for i in range(0, image_values.shape[0]):
            image_values[i] = _normalized_decomposition(image_values[i], normalized)
        for i in range(0, image_values.shape[1]):
            image_values[:, i] = _normalized_decomposition(image_values[:, i], normalized)
        return image_values

    #nonstandard
    else:
        mindim=_read_min_dim(image_values)
        image_values=np.divide(image_values,mindim)
        until=mindim
        while(until>=2):
            for i in range(0, int(until)):
                image_values[i] = _normalized_decomposition_step(image_values[i], until, normalized)
            for i in range(0, int(until)):
                image_values[:, i] = _normalized_decomposition_step(image_values[:, i], until, normalized)
            until/=2
        return image_values


def reconstruction_2d(image_values, normalized=True, standard=True):
    # Haar reconstruction of a 2D array inplace
    #print(image_values.shape)

    if standard:
        for i in range(0, image_values.shape[1]):
            image_values[:, i] = _normalized_reconstruction(image_values[:, i], normalized)
        for i in range(0, image_values.shape[0]):
            image_values[i] = _normalized_reconstruction(image_values[i], normalized)
        return image_values

    #nonstandard
    else:
        mindim=_read_min_dim(image_values)
        until=2
        while(until<=mindim):
            for i in range(0, int(until)):
                image_values[:, i] = _normalized_reconstruction_step(image_values[:, i], until, normalized)
            for i in range(0, int(until)):
                image_values[i] = _normalized_reconstruction_step(image_values[i], until, normalized)
            until*=2
        image_values = np.multiply(image_values, mindim)
        return image_values


def _read_min_dim(image_values):
    if image_values.shape[0] < image_values.shape[1]:
        return image_values.shape[0]
    else:
        return image_values.shape[1]

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
        val1=coefficients[2*i]
        val2=coefficients[2*i+1]
        avg=(val1+val2)*div_sqrt_2
        detail=(val1-val2)*div_sqrt_2
        copy[i]=avg
        copy[detail_i+i]=detail

        global mini
        global maxi
        if avg<mini:mini=avg
        if detail<mini:mini=detail
        if avg>maxi:maxi=avg
        if detail>maxi:maxi=detail

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
        avg = coefficients[i]
        detail = coefficients[detail_index+i]
        val1 = (avg+detail)*div_sqrt_2
        val2 = (avg-detail)*div_sqrt_2
        copy[2*i] = val1
        copy[2*i+1] = val2

    # apply changes to original array
    coefficients = copy
    return coefficients





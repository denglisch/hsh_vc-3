# Visual Computing: Wavelets for Computer Graphics
# team01

import numpy as np
from PIL import Image
import math

mini=0
maxi=0
def kd_test_decomp_recon_on_1d_array():
    c=np.array([9.,7.,2.,6.])
    print("decomp {}".format(c))
    c=_decomposition(c, normalized=False)
    print("recon {}".format(c))
    c=_reconstruction(c,  normalized=False)
    print("-> {}".format(c))
    return


def kd_test_decomp_recon_on_image():
    # load lenna as 2d grayscale array
    image_values = np.array(Image.open('img/Lenna.png').convert('L'))

    # make sure, values can be negative (default uint)
    image_values=image_values.astype(np.float64)

    # std decomp
    print("decomp")
    global mini
    mini=0
    global maxi
    maxi=0
    image_values=decomposition_2d(image_values, normalized=True, standard=False)

    # save img
    #image_values=np.add(image_values, 100.0)
    copy=np.copy(image_values)
    copy=np.multiply(copy, _read_min_dim(image_values))
    #copy=np.add(copy, 128.0)
    copy=copy.astype(np.uint8)
    im = Image.fromarray(copy)
    im.save("img/Lenna_DECOMP.png")
    print("min: {} max: {}".format(mini,maxi))

    #image_values=np.subtract(image_values, 100.0)

    # std recon
    print("recon")
    image_values=reconstruction_2d(image_values, normalized=True, standard=False)

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
    #image_values=compression(image_values, squared_error_stollnitz=20000)
    image_values=compression(image_values, number_of_coeffs_left=10000)
    #image_values=compression(image_values, squared_error=20000000)

    # save img
    image_values=image_values.astype(np.uint8)
    im = Image.fromarray(image_values)
    im.save("img/Lenna_COMP.png")
    return


def kd_test_color_compression_on_yuv_image():
    # load lenna as 2d RGB array
    image_values = np.array(Image.open('img/Lenna.png'))
    print(image_values.shape)
    print(image_values[0,0])

    # convert to YUV
    image_values=RGB2YUV(image_values)
    print(image_values[0,0])

    # decomp and recon is done in compression function
    # compress luminance (Y) with one error
    print("compress Y")
    image_values[:,:,0]=compression(image_values[:,:,0], number_of_coeffs_left=100000)
    # compress chrominance (UV) with another error
    print("compress U")
    image_values[:,:,1]=compression(image_values[:,:,1], number_of_coeffs_left=10000)
    print("compress V")
    image_values[:,:,2]=compression(image_values[:,:,2], number_of_coeffs_left=10000)

    # convert to RGB
    image_values=YUV2RGB(image_values)

    # save img
    image_values=image_values.astype(np.uint8)
    im = Image.fromarray(image_values)
    im.save("img/Lenna_YUV_COMP.png")
    return

# from: https://gist.github.com/Quasimondo/c3590226c924a06b276d606f4f189639
def RGB2YUV(rgb):
    m = np.array([[0.29900, -0.16874, 0.50000],
                  [0.58700, -0.33126, -0.41869],
                  [0.11400, 0.50000, -0.08131]])

    yuv = np.dot(rgb, m)
    yuv[:, :, 1:] += 128.0
    return yuv
# from: https://gist.github.com/Quasimondo/c3590226c924a06b276d606f4f189639
def YUV2RGB(yuv):
    m = np.array([[1.0, 1.0, 1.0],
                  [-0.000007154783816076815, -0.3441331386566162, 1.7720025777816772],
                  [1.4019975662231445, -0.7141380310058594, 0.00001542569043522235]])

    rgb = np.dot(yuv, m)
    rgb[:, :, 0] -= 179.45477266423404
    rgb[:, :, 1] += 135.45870971679688
    rgb[:, :, 2] -= 226.8183044444304
    rgb = rgb.clip(0, 255)
    return rgb

def compression(image_values, squared_error=None, squared_error_stollnitz=None, number_of_coeffs_left=None):
    # return compressed image_values

    # normalized 2d decomp.
    print("- decomposition")
    image_values=decomposition_2d(image_values, normalized=True, standard=True)
    # one array of coefficients
    coefficients=image_values.flatten()
    # sort by decreasing magnitude
    # commented since it can become very slow
    # coefficients[::-1].sort()

    if squared_error is not None:
        # either by max error
        thres_min=np.abs(coefficients).min() # abs(coefficients[len(coefficients)-1])
        thres_max=np.abs(coefficients).max() # abs(coefficients[0])
        threshold = (thres_min + thres_max) / 2.
        epsilon_squared=squared_error
        sum_squared = 0
        # find threashold
        print("- find threshold by max squared error of {}".format(epsilon_squared))
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
            #print("min: {} max: {}".format(thres_min, thres_max))
            found_thres=(thres_max-thres_min)<0.1
        print("-- threshold: {} with squared error of {}".format(threshold, sum_squared))

    #Stollnitz et al.
    # "Another efficient approach to L_2􏰪 compression is to repeatedly
    # discard all coefficients smaller in magnitude than a threshold􏰥
    # increasing the threshold by a factor of two in each iteration􏰥
    # until the allowable level of error is achieved"
    if squared_error_stollnitz is not None:
        print("- find threshold by max squared error of {} (Stollnitz et al.)".format(squared_error_stollnitz))
        # make sure min is not zero
        threshold=np.abs(coefficients[np.nonzero(coefficients)]).min()
        sum_squared = 0
        last_sum_squared=0
        found_thres=False
        while True:
            #print(threshold)
            sum_squared = 0
            for i in range(0, coefficients.size):
                if abs(coefficients[i]) < threshold:
                    sum_squared += math.pow(coefficients[i], 2)
            if sum_squared<=squared_error_stollnitz:
                threshold*=2
                last_sum_squared=sum_squared
            else:
                break
        print("-- threshold: {} with squared error of {}".format(threshold, last_sum_squared))
        print("-- for threshold*2={} squared error would be {}".format(threshold*2, sum_squared))

    # or max coefficients
    # returning threshold also?
    if number_of_coeffs_left is not None:
        print("-number of coeffs: {}".format(number_of_coeffs_left))
        # make sure min is not zero
        threshold=np.abs(coefficients[np.nonzero(coefficients)]).min()
        to_truncate=coefficients.size-number_of_coeffs_left
        if(to_truncate<0):
            print("-- all coefficients will stay")
        else:
            quit_loop=False
            count=0
            count=coefficients.size-np.count_nonzero(image_values)
            while True:
                threshold*=2
                #print(threshold)
                image_values, truncated=truncate_elements_abs_below_threshold(image_values, threshold)
                count+=truncated
                #print("count: {}, threshold: {}".format(count, threshold))
                if count>=to_truncate:
                    break
                number_of_coeffs_left=np.count_nonzero(image_values)
            print("-- truncated {} values ({} %), {} stay ({} %)".format(count, (count/coefficients.size*100), number_of_coeffs_left, number_of_coeffs_left/coefficients.size*100))

    else:
        #only if not number of coeffs
        # truncate small values in image_values
        print("- truncate values smaller than threshold")
        count=0
        for (row, col), value in np.ndenumerate(image_values):
            if abs(value)<threshold:
                image_values[row, col]=0
                count+=1
        print("-- truncated {} values ({} %)".format(count, (count/coefficients.size*100)))

    # recon image
    print("- reconstruction")
    image_values=reconstruction_2d(image_values, normalized=True, standard=True)

    # after return
    # save image
    return image_values

def truncate_elements_abs_below_threshold(image_values, threshold):
    before = np.count_nonzero(image_values)
    image_values[np.abs(image_values) < threshold] = 0
    after = np.count_nonzero(image_values)
    truncated = before - after
    return image_values, truncated


def decomposition_2d(image_values, normalized=True, standard=True):
    # Haar decomposition of a 2D array inplace
    #print(image_values.shape)
    if standard:
        for i in range(0, image_values.shape[0]):
            image_values[i] = _decomposition(image_values[i], normalized)
        for i in range(0, image_values.shape[1]):
            image_values[:, i] = _decomposition(image_values[:, i], normalized)
        return image_values

    #nonstandard
    else:
        mindim=_read_min_dim(image_values)
        image_values=np.divide(image_values,mindim)
        until=mindim
        while(until>=2):
            for i in range(0, int(until)):
                image_values[i] = _decomposition_step(image_values[i], until, normalized)
            for i in range(0, int(until)):
                image_values[:, i] = _decomposition_step(image_values[:, i], until, normalized)
            until/=2
        return image_values


def reconstruction_2d(image_values, normalized=True, standard=True):
    # Haar reconstruction of a 2D array inplace
    #print(image_values.shape)

    if standard:
        for i in range(0, image_values.shape[1]):
            image_values[:, i] = _reconstruction(image_values[:, i], normalized)
        for i in range(0, image_values.shape[0]):
            image_values[i] = _reconstruction(image_values[i], normalized)
        return image_values

    #nonstandard
    else:
        mindim=_read_min_dim(image_values)
        until=2
        while(until<=mindim):
            for i in range(0, int(until)):
                image_values[:, i] = _reconstruction_step(image_values[:, i], until, normalized)
            for i in range(0, int(until)):
                image_values[i] = _reconstruction_step(image_values[i], until, normalized)
            until*=2
        image_values = np.multiply(image_values, mindim)
        return image_values


def _read_min_dim(image_values):
    if image_values.shape[0] < image_values.shape[1]:
        return image_values.shape[0]
    else:
        return image_values.shape[1]

def _decomposition(coefficients, normalized=True):
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
        coefficients=_decomposition_step(coefficients, until, normalized)
        until/=2

    return coefficients

def _decomposition_step(coefficients, until, normalized=True):
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

def _reconstruction(coefficients, normalized=True):
    # Haar reconstruction of array inplace

    # start with 2 and go up to whole size
    until = 2
    while until <= coefficients.size:
        #print(coefficients)
        coefficients = _reconstruction_step(coefficients, until, normalized)
        until *= 2

    # undo normalize
    if normalized:
        sqrt2 = math.sqrt(2)
        np.multiply(coefficients, sqrt2)

    return coefficients

def _reconstruction_step(coefficients, until, normalized=True):
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





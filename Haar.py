# Visual Computing: Wavelets for Computer Graphics
# team01

import numpy as np
from PIL import Image
import math
import util

def kd_test_decomp_recon_on_1d_array():
    c=np.array([9.,7.,2.,6.])
    print("decomp {}".format(c))
    c=_decomposition(c, normalized=False)
    print("-> {}".format(c))
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
    normalized=True
    std=True

    image_values=decomposition_2d(image_values, normalized=normalized, standard=std)

    decomp_image_values=prepare_decomp_image_for_render(image_values, normalized, std)

    im = Image.fromarray(decomp_image_values)
    im.save("img/Lenna_DECOMP.png")

    #image_values=np.subtract(image_values, 100.0)

    # std recon
    print("recon")
    image_values=reconstruction_2d(image_values, normalized=normalized, standard=std)

    # save img
    image_values=image_values.astype(np.uint8)
    im = Image.fromarray(image_values)
    im.save("img/Lenna_RECON.png")
    return

def prepare_decomp_image_for_render(image_values, normalized, std, crop_min_max=True, decomp_img=True):
    color=True if len(image_values.shape)==3 else False
    if color:
        until=3
    else:
        until=1

    # work on copy
    decomp_image_values = np.copy(image_values)

    for i in range(0, until):
        real_extrem = np.array([image_values[i].min(), image_values[i].max()])
        print("- min: {}, max: {}".format(real_extrem[0], real_extrem[1]))
        if crop_min_max:
            extrem = np.array([np.percentile(image_values[i], 1), np.percentile(image_values[i], 99)])
            print("- crop 1% to min: {}, max: {}".format(extrem[0], extrem[1]))
        else:
            extrem = real_extrem

        if decomp_img:
            if normalized:
                sqrt2 = math.sqrt(2)
                if color:
                    decomp_image_values[:,:,i] = np.multiply(decomp_image_values[:,:,i], sqrt2)
                else:
                    decomp_image_values = np.multiply(decomp_image_values, sqrt2)
                extrem *= sqrt2
            if not std:
                min_dim = _read_min_dim(image_values)
                if color:
                    decomp_image_values[:,:,i] = np.multiply(decomp_image_values[:,:,i], min_dim)
                else:
                    decomp_image_values = np.multiply(decomp_image_values, min_dim)
                extrem *= min_dim
            print("- undo normalization/standardization to min: {}, max: {}".format(extrem[0], extrem[1]))

        if crop_min_max:
            decomp_image_values[:,:,i] = np.subtract(decomp_image_values[:,:,i], extrem[0])
            extrem -= extrem[0]
            print("- shift by min to min: {}, max: {}".format(extrem[0], extrem[1]))

            range_ex = extrem[1] - extrem[0]
            correct = 256.0 / range_ex
            decomp_image_values[:,:,i] = np.multiply(decomp_image_values[:,:,i], correct)
            extrem *= correct
            print("- scale to min: {}, max: {}".format(extrem[0], extrem[1]))

    if color:# and decomp_img:
        decomp_image_values = util.YUV2RGB(decomp_image_values).astype(np.uint8)
    else:
        decomp_image_values = decomp_image_values.astype(np.uint8)
    return decomp_image_values


def kd_test_compression_on_image():
    # load lenna as 2d grayscale array
    image_values = np.array(Image.open('img/Lenna.png').convert('L'))

    # make sure, values can be negative (default uint)
    image_values=image_values.astype(np.float64)

    print("compression")
    #image_values=compression(image_values, squared_error_stollnitz=20000)
    #image_values=compression_2d(image_values, number_of_coeffs_left=10000)
    image_values=compression_2d(image_values, number_of_coeffs_left_exact=10000)
    #image_values=compression(image_values, squared_error=20000000)

    # save img
    image_values=image_values.astype(np.uint8)
    im = Image.fromarray(image_values)
    im.save("img/Lenna_COMP.png")
    return


def kd_test_color_compression_on_yuv_image():
    # load lenna as 2d RGB array
    image_values = np.array(Image.open('img/Lenna_1024.jpg'))
    print(image_values.shape)
    print(image_values[0,0])

    # convert to YUV
    image_values=util.RGB2YUV(image_values)
    print(image_values[0,0])

    # decomp and recon is done in compression function
    # compress luminance (Y) with one error
    compression_2d_yuv(image_values, y_number_of_coeffs_left_exact=100, uv_number_of_coeffs_left_exact=50)
    #compression_2d_yuv(image_values, y_number_of_coeffs_left=100, uv_number_of_coeffs_left=50)

    # convert to RGB
    image_values=util.YUV2RGB(image_values)

    # save img
    image_values=image_values.astype(np.uint8)
    Image.fromarray(image_values).save("img/Lenna_1024_YUV_COMP.png")
    return

def kd_test_preprocessing_on_gray():
    # load lenna as 2d grayscale array
    image_values = np.array(Image.open('img/Lenna_WRONG.png').convert('L'))

    # make sure, values can be negative (default uint)
    image_values = image_values.astype(np.float64)

    print("compression")
    # image_values=_compression_2d_only_decomp(image_values, squared_error_stollnitz=20000)
    # image_values=_compression_2d_only_decomp(image_values, number_of_coeffs_left=10000)
    image_values = _compression_2d_only_decomp(image_values, number_of_coeffs_left_exact=60)
    # image_values=_compression_2d_only_decomp(image_values, squared_error=20000000)

    print("- quantization")
    #image_values[image_values<0]=-1
    #image_values[image_values>=0]=1

    print("- reconstruction")
    image_values=reconstruction_2d(image_values, normalized=True, standard=True)

    # save img
    image_values=image_values.astype(np.uint8)

    image_values[image_values < 127] = 0
    image_values[image_values >= 127] = 255

    Image.fromarray(image_values).save("img/Lenna_PRE.png")
    return

def compression_2d_yuv(image_values,
                       y_squared_error=None, y_squared_error_stollnitz=None,
                       y_number_of_coeffs_left=None, y_number_of_coeffs_left_exact=None,
                       uv_squared_error=None, uv_squared_error_stollnitz=None,
                       uv_number_of_coeffs_left=None, uv_number_of_coeffs_left_exact=None):
    print("compress Y")
    image_values[:,:,0]=compression_2d(image_values[:, :, 0],
                                       squared_error=y_squared_error,
                                       squared_error_stollnitz=y_squared_error_stollnitz,
                                       number_of_coeffs_left_exact=y_number_of_coeffs_left_exact,
                                       number_of_coeffs_left=y_number_of_coeffs_left)
    # compress chrominance (UV) with another error
    print("compress U")
    image_values[:,:,1]=compression_2d(image_values[:, :, 1],
                                       squared_error=uv_squared_error,
                                       squared_error_stollnitz=uv_squared_error_stollnitz,
                                       number_of_coeffs_left_exact=uv_number_of_coeffs_left_exact,
                                       number_of_coeffs_left=uv_number_of_coeffs_left)
    print("compress V")
    image_values[:,:,2]=compression_2d(image_values[:, :, 2],
                                       squared_error=uv_squared_error,
                                       squared_error_stollnitz=uv_squared_error_stollnitz,
                                       number_of_coeffs_left_exact=uv_number_of_coeffs_left_exact,
                                       number_of_coeffs_left=uv_number_of_coeffs_left)

    return

def _compression_2d_only_decomp(image_values,
                   squared_error=None,
                   squared_error_stollnitz=None,
                   number_of_coeffs_left_exact=None,
                   number_of_coeffs_left=None):
    # normalized 2d decomp.
    print("- decomposition")
    image_values = decomposition_2d(image_values, normalized=True, standard=True)
    # one array of coefficients
    coefficients = image_values.flatten()
    # sort by decreasing magnitude
    # commented since it can become very slow
    # coefficients[::-1].sort()
    truncated = 0
    threshold = 0
    count_zero_pre = coefficients.size - np.count_nonzero(image_values)
    c_sum = coefficients.size

    if squared_error is not None:
        # either by max error
        print("- find threshold by max squared error of {} (Allerkamp)".format(squared_error))
        thres_min = np.abs(coefficients).min()  # abs(coefficients[len(coefficients)-1])
        thres_max = np.abs(coefficients).max()  # abs(coefficients[0])
        threshold = (thres_min + thres_max) / 2.
        epsilon_squared = squared_error
        sum_squared = 0
        # find threashold
        print("- find threshold by max squared error of {}".format(epsilon_squared))
        found_thres = False
        while not found_thres:
            threshold = (thres_min + thres_max) / 2.
            sum_squared = 0
            for i in range(0, coefficients.size):
                if abs(coefficients[i]) < threshold:
                    sum_squared += math.pow(coefficients[i], 2)
            if sum_squared < epsilon_squared:
                thres_min = threshold
            else:
                thres_max = threshold
            # until thres_min=thres_max
            # print("min: {} max: {}".format(thres_min, thres_max))
            found_thres = (thres_max - thres_min) < 0.1
        print("-- threshold: {} with squared error of {}".format(threshold, sum_squared))

    # Stollnitz et al.
    # "Another efficient approach to L_2􏰪 compression is to repeatedly
    # discard all coefficients smaller in magnitude than a threshold􏰥
    # increasing the threshold by a factor of two in each iteration􏰥
    # until the allowable level of error is achieved"
    if squared_error_stollnitz is not None:
        print("- find threshold by max squared error of {} (Stollnitz et al.)".format(squared_error_stollnitz))
        # make sure min is not zero
        threshold = np.abs(coefficients[np.nonzero(coefficients)]).min()
        sum_squared = 0
        last_sum_squared = 0
        found_thres = False
        while True:
            # print(threshold)
            sum_squared = 0
            for i in range(0, coefficients.size):
                if abs(coefficients[i]) < threshold:
                    sum_squared += math.pow(coefficients[i], 2)
            if sum_squared <= squared_error_stollnitz:
                threshold *= 2
                last_sum_squared = sum_squared
            else:
                break
        print("-- threshold: {} with squared error of {}".format(threshold, last_sum_squared))
        print("-- for threshold*2={} squared error would be {}".format(threshold * 2, sum_squared))

    # or max coefficients
    # returning threshold also?
    if number_of_coeffs_left_exact is not None:
        print("- find threshold by {} coefficients left".format(number_of_coeffs_left_exact))
        # find n-th highest value
        print("-- find {}-th highest magnitude".format(number_of_coeffs_left_exact))
        n = number_of_coeffs_left_exact
        abs_coeff = np.abs(coefficients)
        k = np.partition(abs_coeff, -n)[-n]
        threshold = np.min(np.abs(k))
        # truncate all below

    # or max coefficients
    # returning threshold also?
    # if number_of_coeffs_left is not None:
    if number_of_coeffs_left is not None:
        print("- find threshold by {} coefficients left (approx. Stollnitz)".format(number_of_coeffs_left))

        # make sure min is not zero
        threshold = np.abs(coefficients[np.nonzero(coefficients)]).min()
        to_truncate = coefficients.size - number_of_coeffs_left
        if (to_truncate < 0):
            print("-- all coefficients will stay")
        else:
            count = count_zero_pre
            while True:
                threshold *= 2
                copy = np.copy(image_values)
                copy, plus = truncate_elements_abs_below_threshold(copy, threshold)
                if count + plus >= to_truncate:
                    print("-- stop: next would truncate down to {} elements".format(c_sum - count - plus))
                    threshold /= 2
                    break

    # truncate small values in image_values
    print("- truncate values smaller than threshold of {}".format(threshold))
    image_values, truncated = truncate_elements_abs_below_threshold(image_values, threshold)

    stay = np.count_nonzero(image_values)
    print("-- truncated {} values ({} %), {} were zero before ({} %), {} stay ({} %)".format(
        truncated, truncated / c_sum * 100,
        count_zero_pre, count_zero_pre / c_sum * 100,
        stay, stay / c_sum * 100))

    return image_values

def compression_2d(image_values,
                   squared_error=None,
                   squared_error_stollnitz=None,
                   number_of_coeffs_left_exact=None,
                   number_of_coeffs_left=None):
    # return compressed image_values
    image_values=_compression_2d_only_decomp(image_values,squared_error,squared_error_stollnitz,number_of_coeffs_left_exact,number_of_coeffs_left)
    # recon image
    print("- reconstruction")
    image_values=reconstruction_2d(image_values, normalized=True, standard=True)

    # after return
    # save image
    return image_values

def truncate_elements_abs_below_threshold(image_values, threshold):
    before = np.count_nonzero(image_values)
    image_values[np.abs(image_values) < threshold] = 0
    #image_values=np.where(np.abs(image_values) < threshold,0,image_values)
    after = np.count_nonzero(image_values)
    truncated = before - after
    return image_values, truncated

def decomposition_2d_with_steps(image_values, normalized=True, standard=True, img_list=None, crop_min_max=True):
    # Only on squared images

    #color_image?
    color=True if len(image_values.shape)==3 else False

    if standard:
        # normalize
        if normalized:
            sqrt2 = math.sqrt(2)
            # sqrt2=1
            np.divide(image_values, sqrt2)

        #start with whole array
        until=_read_min_dim(image_values)
        while until>=2:
            for i in range(0, image_values.shape[0]):
                if color:
                    image_values[i,:,0] = _decomposition_step(image_values[i,:,0], until, normalized)
                    image_values[i,:,1] = _decomposition_step(image_values[i,:,1], until, normalized)
                    image_values[i,:,2] = _decomposition_step(image_values[i,:,2], until, normalized)
                else:
                    image_values[i] = _decomposition_step(image_values[i], until, normalized)
            img_list.append(np.copy(prepare_decomp_image_for_render(image_values, normalized, standard, crop_min_max=crop_min_max)))
            until/=2

        img_list.append(None)
        img_list.append(None)

        # start with whole array
        until = _read_min_dim(image_values)
        while until >= 2:
            for i in range(0, image_values.shape[1]):
                if color:
                    image_values[:,i,0] = _decomposition_step(image_values[:,i,0], until, normalized)
                    image_values[:,i,1] = _decomposition_step(image_values[:,i,1], until, normalized)
                    image_values[:,i,2] = _decomposition_step(image_values[:,i,2], until, normalized)
                else:
                    image_values[:, i] = _decomposition_step(image_values[:, i], until, normalized)

            img_list.append(np.copy(prepare_decomp_image_for_render(image_values, normalized, standard, crop_min_max=crop_min_max)))
            until/=2

    #non-standard
    else:
        mindim = _read_min_dim(image_values)
        log=math.ceil(math.log(mindim,2))
        #print(log)
        image_values = np.divide(image_values, mindim)
        until = mindim
        while (until >= 2):
            for i in range(0, int(until)):
                if color:
                    image_values[i,:,0] = _decomposition_step(image_values[i,:,0], until, normalized)
                    image_values[i,:,1] = _decomposition_step(image_values[i,:,1], until, normalized)
                    image_values[i,:,2] = _decomposition_step(image_values[i,:,2], until, normalized)
                else:
                    image_values[i] = _decomposition_step(image_values[i], until, normalized)
            img_list.append(np.copy(prepare_decomp_image_for_render(image_values, normalized, standard, crop_min_max=crop_min_max)))

            if (len(img_list)+1)%int(log+2)==0:
                img_list.append(None)
                img_list.append(None)

            for i in range(0, int(until)):
                if color:
                    image_values[:,i,0] = _decomposition_step(image_values[:,i,0], until, normalized)
                    image_values[:,i,1] = _decomposition_step(image_values[:,i,1], until, normalized)
                    image_values[:,i,2] = _decomposition_step(image_values[:,i,2], until, normalized)
                else:
                    image_values[:, i] = _decomposition_step(image_values[:, i], until, normalized)
            img_list.append(np.copy(prepare_decomp_image_for_render(image_values, normalized, standard,crop_min_max=crop_min_max)))
            until /= 2

    img_list.append(None)
    img_list.append(None)

    return image_values

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
        while until>=2:
            for i in range(0, int(until)):
                image_values[i] = _decomposition_step(image_values[i], until, normalized)
            for i in range(0, int(until)):
                image_values[:, i] = _decomposition_step(image_values[:, i], until, normalized)
            until/=2
        return image_values

def reconstruction_2d_with_steps(image_values, normalized=True, standard=True, img_list=None, crop_min_max=True):
    # Only on squared images

    #color_image?
    color=True if len(image_values.shape)==3 else False

    if standard:
        # normalize
        #if normalized:
        #    sqrt2 = math.sqrt(2)
        #    # sqrt2=1
        #    np.divide(image_values, sqrt2)

        mindim = _read_min_dim(image_values)

        # start with whole array
        until = 2
        while until<=mindim:
            for i in range(0, image_values.shape[1]):
                if color:
                    image_values[:,i,0] = _reconstruction_step(image_values[:,i,0], until, normalized)
                    image_values[:,i,1] = _reconstruction_step(image_values[:,i,1], until, normalized)
                    image_values[:,i,2] = _reconstruction_step(image_values[:,i,2], until, normalized)
                else:
                    image_values[:, i] = _reconstruction_step(image_values[:, i], until, normalized)
            img_list.append(np.copy(prepare_decomp_image_for_render(image_values, normalized, standard, crop_min_max)))
            until*=2

        img_list.append(None)
        img_list.append(None)

        #start with whole array
        until=2
        while until<=mindim:
            for i in range(0, image_values.shape[0]):
                if color:
                    image_values[i,:,0] = _reconstruction_step(image_values[i,:,0], until, normalized)
                    image_values[i,:,1] = _reconstruction_step(image_values[i,:,1], until, normalized)
                    image_values[i,:,2] = _reconstruction_step(image_values[i,:,2], until, normalized)
                else:
                    image_values[i] = _reconstruction_step(image_values[i], until, normalized)
            img_list.append(np.copy(prepare_decomp_image_for_render(image_values, normalized, standard,crop_min_max)))
            until*=2

        # undo normalize
        if normalized:
            sqrt2 = math.sqrt(2)
            np.multiply(image_values, sqrt2)

    #non-standard
    else:
        mindim = _read_min_dim(image_values)
        log=math.ceil(math.log(mindim,2))
        #print(log)
        until = 2
        while until<=mindim:
            for i in range(0, int(until)):
                if color:
                    image_values[:,i,0] = _reconstruction_step(image_values[:,i,0], until, normalized)
                    image_values[:,i,1] = _reconstruction_step(image_values[:,i,1], until, normalized)
                    image_values[:,i,2] = _reconstruction_step(image_values[:,i,2], until, normalized)
                else:
                    image_values[:, i] = _reconstruction_step(image_values[:, i], until, normalized)
            img_list.append(np.copy(prepare_decomp_image_for_render(image_values, normalized, standard,crop_min_max)))

            if (len(img_list)+1)%int(log+2)==0:
                img_list.append(None)
                img_list.append(None)

            for i in range(0, int(until)):
                if color:
                    image_values[i,:,0] = _reconstruction_step(image_values[i,:,0], until, normalized)
                    image_values[i,:,1] = _reconstruction_step(image_values[i,:,1], until, normalized)
                    image_values[i,:,2] = _reconstruction_step(image_values[i,:,2], until, normalized)
                else:
                    image_values[i] = _reconstruction_step(image_values[i], until, normalized)
            img_list.append(np.copy(prepare_decomp_image_for_render(image_values, normalized, standard,crop_min_max)))
            until *= 2

        image_values = np.multiply(image_values, mindim)

    #correct last image
    img_list[len(img_list)-1]=np.copy(prepare_decomp_image_for_render(image_values, normalized, standard,crop_min_max, False))

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





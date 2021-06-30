# Visual Computing: Wavelets for Computer Graphics
# team01

import Haar as Haar
import util
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import math

def main():
    #exit(0)


    color=True
    #decomp/recon
    norm = True
    std = True
    #postprocessing
    crop_min_max = False

    image_pathes=["img/Lenna.png",
                  "img/Mona_Bean.png"]

    # 1 & 2 Haar trafo
    if True:
        Haar.kd_test_decomp_recon_on_1d_array()
        decomp_and_recon(image_pathes[1], color=True, normalized=True, standard=False)
        #decomp_and_recon_with_steps(image_pathes[1], color=False, normalized=True, standard=False, crop_min_max=False)
        #decomp_and_recon_with_steps(image_pathes[1], color=True, normalized=False, standard=True, crop_min_max=True)
        #Green effect:
        decomp_and_recon_with_steps(image_pathes[1], color=True, normalized=True, standard=True, crop_min_max=False)

    # 3 compression gray
    if False:
        infile=image_pathes[1]
        image_values=load_image_as_ndarray_float(infile, gray=True)
        print("compression")
        #image_values=compression(image_values, squared_error=20000000)
        #image_values=compression(image_values, squared_error_stollnitz=20000)
        #image_values=compression_2d(image_values, number_of_coeffs_left=10000)
        image_values=Haar.compression_2d(image_values, number_of_coeffs_left_exact=10000)
        outfile=infile+"_GRAY_COMP.png"
        save_as_image(image_values, outfile)

    # 4 compression yuv
    if False:
        infile=image_pathes[1]
        image_values=load_image_as_ndarray_float(infile, gray=False, yuv=True)
        print("compression")
        #total values 512*512*3=786.432
        #per channel: 262.144
        #image_values=Haar.compression_2d_yuv(image_values, y_number_of_coeffs_left_exact=2621, uv_number_of_coeffs_left_exact=1310)
        image_values=Haar.compression_2d_yuv(image_values, y_number_of_coeffs_left_exact=26214, uv_number_of_coeffs_left_exact=13107)
        #image_values=Haar.compression_2d_yuv(image_values, y_number_of_coeffs_left=100, uv_number_of_coeffs_left=50)
        #image_values=Haar.compression_2d_yuv(image_values, y_squared_error=10000, uv_squared_error=5000)
        #image_values=Haar.compression_2d_yuv(image_values, y_squared_error_stollnitz=10000, uv_squared_error_stollnitz=5000)
        outfile=infile+"_YUV_COMP.png"
        save_as_image(image_values, outfile, yuv=True)

    # 5 IQA Preprocessing
    if False:
        #gray
        preprocessing(image_pathes[1],number_of_coeffs_left_exact=60)
        #TODO If neccessary...
        #color
        #preprocessing(image_pathes[1],gray=False, yuv=True, number_of_coeffs_left_exact=60)
    return

def preprocessing(image_path, gray=True, yuv=False, number_of_coeffs_left_exact=60):
    """
    Performs image query preprocessing by compressing image down to number_of_coeffs_left_exact an do quantization
    :param image_path: image to load
    :param gray and yuy: load and perform as gray or yuv image
    :param number_of_coeffs_left_exact: number of coefficiants that are not truncated in compression step
    """
    image_values = load_image_as_ndarray_float(image_path, gray=gray, yuv=yuv)
    print("compression")
    image_values = Haar.compression_2d_only_decomp(image_values, number_of_coeffs_left_exact=number_of_coeffs_left_exact)

    # TODO: HERE OR THERE?
    print("quantization")
    # image_values[image_values<0]=-1
    # image_values[image_values>=0]=1

    print("reconstruction")
    image_values = Haar.reconstruction_2d(image_values, normalized=True, standard=True)

    # TODO: HERE OR THERE?
    image_values[image_values < 127] = 0
    image_values[image_values >= 127] = 255

    outfile=image_path
    if gray:
        outfile+="_GRAY"
    else:
        outfile+="_YUV"
    outfile+="_PRE.png"
    save_as_image(image_values, outfile, yuv)


def decomp_and_recon(image_path, color, normalized, standard):
    """
    Perform Haar decomposition and reconstruction and saves results as images
    :param image_path: image to load
    :param color: load and perform as gray or yuv image
    :param normalized: normalized Haar trafo
    :param standard: standard or non-standard Haar trafo
    """
    yuv=True if color else False
    gray=False if color else True

    image_values=load_image_as_ndarray_float(image_path, gray=gray, yuv=yuv)

    if color:
        image_values[:, :, 0] = Haar.decomposition_2d(image_values[:, :, 0], normalized=normalized, standard=standard)
        image_values[:, :, 1] = Haar.decomposition_2d(image_values[:, :, 1], normalized=normalized, standard=standard)
        image_values[:, :, 2] = Haar.decomposition_2d(image_values[:, :, 2], normalized=normalized, standard=standard)

    else:
        image_values=Haar.decomposition_2d(image_values, normalized=normalized, standard=standard)

    outfile=image_path
    if gray:
        outfile+="_GRAY"
    else:
        outfile+="_YUV"
    outfile+="_DECOMP.png"
    save_as_image(Haar.prepare_decomp_image_for_render(image_values, normalized=normalized, std=standard, decomp_img=True), outfile, yuv=yuv)

    if color:
        image_values[:, :, 0] = Haar.reconstruction_2d(image_values[:, :, 0], normalized=normalized, standard=standard)
        image_values[:, :, 1] = Haar.reconstruction_2d(image_values[:, :, 1], normalized=normalized, standard=standard)
        image_values[:, :, 2] = Haar.reconstruction_2d(image_values[:, :, 2], normalized=normalized, standard=standard)
    else:
        image_values = Haar.reconstruction_2d(image_values, normalized=normalized, standard=standard)

    outfile=image_path
    if gray:
        outfile+="_GRAY"
    else:
        outfile+="_YUV"
    outfile+="_RECON.png"
    save_as_image(image_values, outfile, yuv=yuv)

def decomp_and_recon_with_steps(image_path, color, normalized, standard, crop_min_max):
    """
    Performs Haar decomposition and reconstruction showing all intermediate steps in grid

    :param image_path: image to process
    :param color: grayscale or yuv processing
    :param normalized: normalized Haar trafo
    :param standard: standard or non-standard Haar trafo
    :param crop_min_max: tells if image will be prepared for visualization only (escp. for normalized trafo. Otherwise nearly all values will be near zero)
    """
    yuv=True if color else False
    gray=False if color else True

    image_values=load_image_as_ndarray_float(image_path, gray=gray, yuv=yuv)
    print("img shape {}".format(image_values.shape))

    img_list = []
    if yuv:
        # img_list.append(np.copy(image_values.astype(np.uint8)))
        img_list.append(np.copy(util.YUV2RGB(image_values).astype(np.uint8)))
    else:
        img_list.append(np.copy(image_values))

    #filling list within these functions
    print("decomp")
    decomp_image_values = Haar.decomposition_2d_with_steps(image_values, normalized=normalized, standard=standard,
                                                           img_list=img_list, crop_min_max=crop_min_max)

    print("recon")
    result = Haar.reconstruction_2d_with_steps(decomp_image_values, normalized=normalized, standard=standard, img_list=img_list,
                                               crop_min_max=crop_min_max)

    #calculating diff-image
    if yuv:
        # diff_img = np.subtract(util.YUV2RGB(image_values), util.YUV2RGB(result))
        diff_img = np.copy(np.subtract(image_values, result))
    else:
        diff_img = np.subtract(image_values, result)
    diff_img = np.abs(diff_img)
    diff_img = diff_img.astype(np.uint8)

    img_list.append(diff_img)
    error = diff_img.sum()
    mul = 3 if color else 1
    rel_error = error / (image_values.shape[0] * image_values.shape[1] * mul * 256) * 100
    print("Diff from input: {} ({} %)".format(error, rel_error))

    # print(len(img_list))
    mindim = Haar.read_min_dim(image_values)
    log = math.ceil(math.log(mindim, 2))

    #finally render
    render_images(img_list, log + 2, 5)

def render_images(img_list, columns=4, rows=2):
    """
    renders images in img_list as grid of cols and rows
    :param img_list: list of 2d numpy.arrays containing image values (in rgb or gray [0,255])
    """
    fig = plt.figure(frameon=False,figsize=(columns,rows))
    #plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=1)

    plt.get_current_fig_manager().set_window_title('Haar Trafo')

    color=True if len(img_list[0].shape)==3 else False


    for i in range(0, len(img_list)):
        if img_list[i] is not None:
            #img_list[i]=util.YUV2RGB(img_list[i]).astype(np.uint8)

            img_ax=fig.add_subplot(rows, columns, i + 1)
            img_ax.set_axis_off()
            if color:
                img_ax.imshow(img_list[i])
            else:
                img_ax.imshow(img_list[i], cmap='gray')

            #titles
            if i==0:
                img_ax.title.set_text('Input')
            if i==1:
                img_ax.title.set_text('Decomp...')
            if i==len(img_list)/2+1:
                img_ax.title.set_text('Recon...')
            if i==len(img_list)-2:
                img_ax.title.set_text('Result')
            if i==len(img_list)-1:
                img_ax.title.set_text('Diff.')
    plt.tight_layout()

    plt.show()

def save_as_image(image_values, image_path, yuv=False):
    """
    :param image_values: as numpy.array
    :param path: output-file
    """
    if yuv:
        image_values=util.YUV2RGB(image_values)

    # save image at path
    image_values=image_values.astype(np.uint8)
    im = Image.fromarray(image_values)
    im.save(image_path)


def load_image_as_ndarray_float(image_path, gray=True, yuv=False):
    """
    :param image_path: image to load
    :param gray: laod as gray or rgb
    :param yuv: convert to yuv
    :return: 2d numpy.array containing image values
    """
    if gray:
        image_values = np.array(Image.open(image_path).convert('L'))
    else:
        image_values = np.array(Image.open(image_path))

    if yuv:
        #reduce to 3 channels (otherwise our conversion won't work ;) )
        image_values=image_values[:,:,:3]
        image_values = image_values.astype(np.float64)
        image_values = util.RGB2YUV(image_values)

    # make sure, values can be negative (default uint)
    image_values=image_values.astype(np.float64)
    return image_values

if __name__ == "__main__":
    # execute only if run as a script
    main()

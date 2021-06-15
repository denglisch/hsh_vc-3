import Haar as Haar
import util
import numpy as np
from PIL import Image
import argparse
import os
import matplotlib.pyplot as plt

#TODO: @Dennis: Zwischenschritte: alle? oder reicht das so?

#TODO CR
# main & params
# load image
#   Info: https://note.nkmk.me/en/python-numpy-image-processing/
#   image_values = np.array(Image.open('img/Lenna.png'))
# convert into grayscale
#    im_gray = np.array(Image.open('img/Lenna.png').convert('L'))
# convert into YUV
#    Haar.RGB2YUV and YUV2RGB
# function calls with params
# render images as figs in plt
#   FUNCTION render 8
# save results as image file save_as_image(image, path)

#TODO KD
# haar 2d
# other stuff ;)
image_path = None
def load_args():
    global image_path
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, default= "Lenna.png", help='image_folder missing')
    parser.add_argument('--g', action="store_true")
    parser.add_argument('--yuv', action="store_true")
    #parser.add_argument('--g', type=int, default=0 )
    args = parser.parse_args()
    cwd = os.getcwd()
    image_path = os.path.join(cwd, "img", args.file)
    print("load file {}".format(image_path))
    return args


def main():
    args = load_args()
    img = Image.open(image_path)
    if args.g:
        img_values = np.array(img.convert('L'))

    if args.yuv:
        img_values = util.RGB2YUV(img)
    else:
        img_values = np.array(img)
    print("img shape {}".format(img_values.shape))

    img_list =[]
    for i in range(0,8):
        if i == 3 or i == 4:
            img_list.append(None)
        else:
            img_list.append(img_values)

    print(len(img_list))
    render_8_images(img_list)
    exit(0)

    Haar.kd_test_decomp_recon_on_1d_array()
    #Haar.kd_test_decomp_recon_on_image()
    #Haar.kd_test_compression_on_image()
    Haar.kd_test_color_compression_on_yuv_image()
    #wirte image_path var
    #TODO: get params
    # -decom. n recon. haar_2d
    #   - std. and non-std.
    # -compress
    #   - either number of coefficiants [1 to max(width, height)]
    #   - or max error [0,1000]
    #   - gray or color
    # -image query preprocess
    #   - decomp. OR recon.

    #load image
    return


def render_images(img_list, columns=4, rows=2):
    fig = plt.figure(figsize=(12, 8))
    for i in range(0, columns * rows):
        if img_list[i] is not None:
            fig.add_subplot(rows, columns, i + 1)
            plt.imshow(img_list[i])
    plt.show()

def render_4_images(img_list):
    render_images(img_list, columns=2, rows=2)


def render_8_images(img_list):
    """
    array element 3 und 4 müssen mit None gefüllt sein um das muster zu erzeugen
    :param img_list:
    :return:
    """
    render_images(img_list)

def save_as_image(image_values, image_path, yuv=False):
    """

    :param image_values: als numpy.array
    :param path:
    :return:
    """
    if yuv:
        image_values=util.YUV2RGB(image_values)

    # save image at path
    image_values=image_values.astype(np.uint8)
    im = Image.fromarray(image_values)
    im.save(image_path)


def load_image_as_ndarray_float(image_path='img/Lenna.png', gray=True, yuv=False):
    if gray:
        image_values = np.array(Image.open(image_path).convert('L'))
    else:
        image_values = np.array(Image.open(image_path))

    if yuv:
        image_values = util.RGB2YUV(image_values)

    # make sure, values can be negative (default uint)
    image_values=image_values.astype(np.float64)
    return image_values


def haar_2d(image, std=True):

    # load image as grayscale
    image="LOL"

    # run Haar decomposition
    #TODO einmal std. einmal nicht-std.
    # Parameters args[]
    # std. or non-std.
    Haar.decomposition_2d(image)
    # save result of decomposition

    # run Haar reconstruction
    Haar.reconstruction_2d(image)
    # save result of reconstruction
    save_as_image(image, image_path+"_decomp_c2.png")

    #TODO show result
    # 4 times for decomp. 4 times for recon. (4x4 images)

    return

def compress(image, c_count=None, max_error=None, color=False):
    #load image as grayscale

    if c_count is not None:
        print()
    if max_error is not None:
        print()

    #TODO: Haar for YUV

    #run Haar decomp
    #run compression with params

    #same as above with recon and show
    return

def image_query_preprocess(image, decom=True):
    # return hard compressed image
    return

if __name__ == "__main__":
    # execute only if run as a script
    main()

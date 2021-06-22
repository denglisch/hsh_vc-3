import Haar as Haar
import util
import numpy as np
from PIL import Image
import argparse
import os
import matplotlib.pyplot as plt
import math

#TODO: @Dennis: Zwischenschritte: alle? oder reicht das so?
# - "lib" kann das soweit, reichen die "test"-Fkt?
# - nur auf shapes zur Basis 2
# - Werte, die bleiben sollen ungefähr ok?
# - Preprocessing: Nur compression auf 60 coefficients?

#TODO CR
# main & params
# function calls with params

#TODO KD
#
image_path = None
def load_args():
    global image_path
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, default="img/Lenna.png", help='image_folder missing')
    parser.add_argument('--gray', action="store_true")
    parser.add_argument('--yuv', action="store_true")

    #TODO:
    # normalized: --norm (default true)
    # standard/ non standard: --non-std (default false)
    # repair render
    #TODO: get params
    # -decom. n recon. haar_2d
    #   - std. and non-std.
    # -compress
    #   - either number of coefficiants [1 to max(width, height)]
    #   - or max error [0,1000]
    #   - gray or color
    # -image query preprocess
    #   - decomp. OR recon.

    #parser.add_argument('--g', type=int, default=0 )
    args = parser.parse_args()
    cwd = os.getcwd()
    #image_path = os.path.join(cwd, "img", args.file)
    image_path=args.file
    print("load file {}".format(image_path))
    return args


def main():
    args = load_args()

    color=True
    yuv=True if color else False
    gray=False if color else True

    #image_values=load_image_as_ndarray_float(image_path, args.gray, args.yuv)
    #image_values=load_image_as_ndarray_float(image_path, gray=gray, yuv=yuv)
    image_values=load_image_as_ndarray_float("img/Lenna.png", gray=gray, yuv=yuv)
    print("img shape {}".format(image_values.shape))

    Haar.kd_test_decomp_recon_on_1d_array()
    #Haar.kd_test_decomp_recon_on_image()
    #Haar.kd_test_compression_on_image()
    #Haar.kd_test_color_compression_on_yuv_image()
    #exit(0)

    img_list=[]
    if yuv:
        #img_list.append(np.copy(image_values.astype(np.uint8)))
        img_list.append(np.copy(util.YUV2RGB(image_values).astype(np.uint8)))
    else:
        img_list.append(np.copy(image_values))


    norm=True
    std=True
    crop_min_max = False
    print("decomp")
    decomp_image_values=Haar.decomposition_2d_with_steps(image_values, normalized=norm, standard=std, img_list=img_list, crop_min_max=crop_min_max)

    print("recon")
    result=Haar.reconstruction_2d_with_steps(decomp_image_values, normalized=norm, standard=std, img_list=img_list, crop_min_max=crop_min_max)

    #save_as_image(result,"img/Lenna_Test.png",True)

    if yuv:
        #diff_img = np.subtract(util.YUV2RGB(image_values), util.YUV2RGB(result))
        diff_img = np.copy(np.subtract(image_values, result))
    else:
        diff_img=np.subtract(image_values,result)
    #diff_img.astype(np.uint8)
    diff_img=Haar.prepare_decomp_image_for_render(diff_img, normalized=norm, std=std,crop_min_max=crop_min_max, decomp_img=False)
    img_list.append(diff_img)
    error=diff_img.sum()
    mul=3 if color else 1
    rel_error=error/(image_values.shape[0]*image_values.shape[1]*mul*256)*100
    print("Diff from input: {} ({} %)".format(error, rel_error))

    #print(len(img_list))
    mindim = Haar._read_min_dim(image_values)
    log = math.ceil(math.log(mindim, 2))

    render_images(img_list,log+2,5)
    #render_8_images(img_list)
    exit(0)


    #wirte image_path var

    #load image
    return


def render_images(img_list, columns=4, rows=2):
    fig = plt.figure(frameon=False)#, figsize=(12,8))
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=1)

    #fig.set_size_inches(1, 2)
    #print(img_list)

    #ax = plt.Axes(fig, [0., 0., 1., 1.])
    #ax.set_axis_off()
    #fig.add_axes(ax)

    color=True if len(img_list[0].shape)==3 else False

    #for i in range(0, columns * rows):
    for i in range(0, len(img_list)):
        if img_list[i] is not None:
            #img_list[i]=util.YUV2RGB(img_list[i])

            pl=fig.add_subplot(rows, columns, i + 1)
            pl.set_axis_off()
            if color:
                plt.imshow(img_list[i])
            else:
                plt.imshow(img_list[i], cmap='gray')

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
        #reduce to 3 channels (otherwise our conversion won't work ;) )
        #image_values=image_values[...,:3]
        image_values=image_values[:,:,:3]
        image_values = image_values.astype(np.float64)
        image_values = util.RGB2YUV(image_values)

    # make sure, values can be negative (default uint)
    image_values=image_values.astype(np.float64)
    #image_values=image_values.astype(np.float32)
    return image_values


def haar_2d(image, std=True):

    # load image as grayscale
    image="LOL"

    # run Haar decomposition
    Haar.decomposition_2d(image)
    # save result of decomposition

    # run Haar reconstruction
    Haar.reconstruction_2d(image)
    # save result of reconstruction
    save_as_image(image, image_path+"_decomp_c2.png")

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

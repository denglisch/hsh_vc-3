import Haar as Haar

#TODO: @Dennis: Zwischenschritte: alle? oder reicht das so?

#TODO CR
# main & params
# load image
#   Info: https://note.nkmk.me/en/python-numpy-image-processing/
#   image_values = np.array(Image.open('img/Lenna.png'))
# convert into grayscale
#    im_gray = np.array(Image.open('img/Lenna.png').convert('L'))
# convert into YUV
#   from skimage.color import rgb2yuv
#   img = Image.open('test.jpeg')
#   img_yuv = rgb2yuv(img)
# function calls with params
# render images as figs in plt
#   FUNCTION render 8
# save results as image file save_as_image(image, path)

#TODO KD
# haar 2d
# other stuff ;)

image_path=None
def main():
    Haar.kd_test_decomp_recon_on_1d_array()
    Haar.kd_test_decomp_recon_on_image()
    Haar.kd_test_compression_on_image()
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

def render_4_images(images_array_of_4):
    return
def render_8_images(images_array_of_8):
    return
def save_as_image(image, path):
    # save image at path
    return

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

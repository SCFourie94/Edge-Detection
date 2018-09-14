###Author: Stephan Fourie

import numpy as np
import math
import cv2
import sys
import scipy.ndimage as im
import scipy.misc as misc
import scipy.signal as signal
import numpy as np
import matplotlib.pyplot as plot

#function to process images
def process_image(name_in, name_out_log, name_out_zero, LoGMask, maskSize):

    # read image
    img_in = im.imread(name_in)

    # Filter
    img_log = convolve(img_in, LoGMask)

    # Zero crossing
    img_zero = zeroCrossing(img_log, 0.04)

    #save image log
    img_plot = plot.imshow(img_log, cmap='gray')
    plot.savefig(name_out_log)

    #save image zero
    img_plot = plot.imshow(img_zero, cmap='gray')
    plot.savefig(name_out_zero)

def drawMasks(mask, mask_out):
    plot.imshow(mask, cmap='Reds')
    plot.colorbar()
    plot.savefig(mask_out)
    plot.show()

# Define LoG equation
def LoG_Formula(x, y, sig):
    return ((np.power(x, 2) + np.power(y, 2) - (2 * np.power(sig, 2))) / np.power(sig, 4)) * np.exp(-1 * ((np.power(x, 2) + np.power(y, 2)) / (2 * np.power(sig, 2))))

# Generate a LoG mask
def gen_LoGMask(maskSize, sig):

    mask = np.zeros((maskSize, maskSize))

    parameter = int((maskSize - 1) / 2)

    for i in range(maskSize):
        for j in range(maskSize):
            mask[i, j] = LoG_Formula(i - parameter, j - parameter, sig)

    return mask

def convolve(image, mask):
    width = image.shape[0]
    height = image.shape[1]
    mask_s = mask.shape[0]
    mask = np.flipud(np.fliplr(mask))
    img_out = np.zeros(image.shape)

    contstraints = int((mask_s - 1)/2)

    #pad image
    padded_img = np.zeros((height + mask_s - 1, width + mask_s - 1))
    padded_img[contstraints:-contstraints, contstraints:-contstraints] = image

    for x in range(height):
        for y in range(width):
            img_out[y,x] = (mask*padded_img[y:y+mask_s,x:x+mask_s]).sum()
            
    return img_out

def zeroCrossing(img_in, constant):
    width = img_in.shape[0]
    height = img_in.shape[1]
    img_out = np.zeros(img_in.shape)

    thresh = float(np.absolute(img_in).max()) * constant

    for x in range(1, height - 1):
        for y in range(1, width - 1):
            pixel = img_in[x,y]
            pad = img_in[x-1:x+2, y-1:y+2]
            max_pad = pad.max()
            min_pad = pad.min()
            
            if np.sign(pixel) > 0 and np.sign(min_pad) < 0:
                zeroCross = True
            elif np.sign(pixel) < 0 and np.sign(max_pad) > 0:
                zeroCross = True
            else:
                zeroCross = False

            if zeroCross and ((max_pad - min_pad) > thresh):
                img_out[x,y] = 1
            else:
                img_out[x,y] = 0

    return img_out

# Filter with LoG mask
def filter(img_in, mask, maskSize):

    img_in = cv2.copyMakeBorder(img_in,maskSize,maskSize,maskSize,maskSize,cv2.BORDER_CONSTANT,value=0)
    
    # Get height and width of image
    height = img_in.shape[0]
    width = img_in.shape[1]

    # Middle of mask
    parameter = int((maskSize-1)/2);

    #create copy of img
    img_out = img_in.copy()

    for i in range(height-maskSize):
        for j in range(width-maskSize):
            sum = 0
            for k in range(maskSize):
                for g in range(maskSize):
                    a = img_in.item(i+k, j+g)
                    p = mask[k, g]
                    sum = sum - (p * a)
            img_out.itemset((i+parameter, j+parameter), sum)
    return img_out

def main(argv):

    LoGmask7 = gen_LoGMask(7, 1)
    drawMasks(LoGmask7, 'Documentation/LoG_Results/LoG_mask7.jpg')
    LoGmask13 = gen_LoGMask(13, 2)
    drawMasks(LoGmask13, 'Documentation/LoG_Results/LoG_mask13.jpg')
    LoGmask25 = gen_LoGMask(25, 4)
    drawMasks(LoGmask25,'Documentation/LoG_Results/LoG_mask25.jpg')

    process_image("lena.png", 'Documentation/LoG_Results/lena7.png', 'Documentation/LoG_Results/lena7_zero.png', LoGmask7, 7)
    process_image("lena.png", 'Documentation/LoG_Results/lena13.png',  'Documentation/LoG_Results/lena13_zero.png', LoGmask13, 13)
    process_image("lena.png", 'Documentation/LoG_Results/lena25.png',  'Documentation/LoG_Results/lena25_zero.png', LoGmask25, 25)

    process_image("cameraman.jpg", 'Documentation/LoG_Results/cam7.jpg', 'Documentation/LoG_Results/cam7_zero.jpg', LoGmask7, 7)
    process_image("cameraman.jpg", 'Documentation/LoG_Results/cam3.jpg',  'Documentation/LoG_Results/cam13_zero.jpg', LoGmask13, 13)
    process_image("cameraman.jpg", 'Documentation/LoG_Results/cam25.jpg',  'Documentation/LoG_Results/cam25_zero.jpg', LoGmask25, 25)

    return 0


if __name__ == "__main__":
    main(sys.argv[1:])

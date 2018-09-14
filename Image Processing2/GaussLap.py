###Author: Stephan Fourie

import numpy as np
import cv2
import sys
import scipy.ndimage as im
import scipy.misc as misc
import scipy.signal as signal
import matplotlib.pyplot as plot
from PIL import Image

#function to process images
def process_image(name_in, name_out_gauss, name_out_lap, name_out_zero, gaussMask, lapMask, maskSize):

    # read image
    img_in = cv2.imread(name_in, 0)

    # Filter with gaussian
    img_gauss = signal.convolve(img_in, gaussMask)
    
    # Filter with laplacian
    img_lap = signal.convolve(img_gauss, lapMask)

    # Zero crossing
    img_zero = zeroCrossing(img_lap, 0.04)

    #save image smooth
    img_plot = plot.imshow(img_gauss, cmap='gray')
    plot.savefig(name_out_gauss)

    #save image lap
    img_plot = plot.imshow(img_lap, cmap='gray')
    plot.savefig(name_out_lap)

    #save image zero
    img_plot = plot.imshow(img_zero, cmap='gray')
    plot.savefig(name_out_zero)

def drawMasks(mask, mask_out):
    plot.imshow(mask, cmap='Blues_r')
    plot.colorbar()
    plot.savefig(mask_out)
    plot.show()

# Define gaussian equation
def gauss_Formula(x, y, sig):
    return -1 * (1/(sig * np.sqrt(2 * np.pi))) * (np.exp(-1 * ((np.power(x, 2) + np.power(y, 2)) / (2 * np.power(sig, 2)))))

# Generate a gaussian mask
def gen_GaussMask(maskSize, sig):

    mask = np.zeros((maskSize, maskSize))

    parameter = int((maskSize - 1) / 2);

    for i in range(maskSize):
        for j in range(maskSize):
            mask[i, j] = gauss_Formula(i - parameter, j - parameter, sig)
    return mask / np.sum(mask)

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

def main(argv):

    gauss_mask7 = gen_GaussMask(7, 1)
    drawMasks(gauss_mask7, 'Documentation/Gaussian_Lap_Results/Gauss_mask7.jpg')
    gauss_mask13 = gen_GaussMask(13, 2)
    drawMasks(gauss_mask13, 'Documentation/Gaussian_Lap_Results/Gauss_mask13.jpg')
    gauss_mask25 = gen_GaussMask(25, 4)
    drawMasks(gauss_mask25,'Documentation/Gaussian_Lap_Results/Gauss_mask25.jpg')

    lap_Mask = np.matrix([[-1, -1, -1],[-1, 8, -1],[-1, -1, -1]])
    drawMasks(lap_Mask, 'Documentation/Gaussian_Lap_Results/Lap_mask.jpg')

    process_image("lena.png", 'Documentation/Gaussian_Lap_Results/lena_7_Gauss.jpg', 'Documentation/Gaussian_Lap_Results/lena_7_LAP.jpg', 'Documentation/Gaussian_Lap_Results/lena_7_ZERO.jpg', gauss_mask7, lap_Mask, 7)
    process_image("lena.png", 'Documentation/Gaussian_Lap_Results/lena_13_Gauss.jpg', 'Documentation/Gaussian_Lap_Results/lena_13_LAP.jpg', 'Documentation/Gaussian_Lap_Results/lena_13_ZERO.jpg', gauss_mask13, lap_Mask, 13)
    process_image("lena.png", 'Documentation/Gaussian_Lap_Results/lena_25_Gauss.jpg', 'Documentation/Gaussian_Lap_Results/lena_25_LAP.jpg', 'Documentation/Gaussian_Lap_Results/lena_25_ZERO.jpg', gauss_mask25, lap_Mask, 25)

    process_image("cameraman.jpg", 'Documentation/Gaussian_Lap_Results/cameraman_7_Gauss.jpg', 'Documentation/Gaussian_Lap_Results/cameraman_7_LAP.jpg', 'Documentation/Gaussian_Lap_Results/cameraman_7_ZERO.jpg', gauss_mask7, lap_Mask, 7)
    process_image("cameraman.jpg", 'Documentation/Gaussian_Lap_Results/cameraman_13_Gauss.jpg', 'Documentation/Gaussian_Lap_Results/cameraman_13_LAP.jpg', 'Documentation/Gaussian_Lap_Results/cameraman_13_ZERO.jpg', gauss_mask13, lap_Mask, 13)
    process_image("cameraman.jpg", 'Documentation/Gaussian_Lap_Results/cameraman_25_Gauss.jpg', 'Documentation/Gaussian_Lap_Results/cameraman_25_LAP.jpg', 'Documentation/Gaussian_Lap_Results/cameraman_25_ZERO.jpg', gauss_mask25, lap_Mask, 25)
    
    process_image("zebras.jpg", 'Documentation/Gaussian_Lap_Results/zebras_7_Gauss.jpg', 'Documentation/Gaussian_Lap_Results/zebras_7_LAP.jpg', 'Documentation/Gaussian_Lap_Results/zebras_7_ZERO.jpg', gauss_mask7, lap_Mask, 7)
    process_image("zebras.jpg", 'Documentation/Gaussian_Lap_Results/zebras_13_Gauss.jpg', 'Documentation/Gaussian_Lap_Results/zebras_13_LAP.jpg', 'Documentation/Gaussian_Lap_Results/zebras_13_ZERO.jpg', gauss_mask13, lap_Mask, 13)
    process_image("zebras.jpg", 'Documentation/Gaussian_Lap_Results/zebras_25_Gauss.jpg', 'Documentation/Gaussian_Lap_Results/zebras_25_LAP.jpg', 'Documentation/Gaussian_Lap_Results/zebras_25_ZERO.jpg', gauss_mask25, lap_Mask, 25)

    return 0


if __name__ == "__main__":
    main(sys.argv[1:])

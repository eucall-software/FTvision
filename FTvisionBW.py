'''
Simply display the contents of the webcam with optional mirroring using OpenCV
via the new Pythonic cv2 interface.  Press <esc> to quit.
'''
from cv2 import VideoCapture
from cv2 import flip
from cv2 import imread, imshow
from cv2 import waitKey
from cv2 import destroyAllWindows
import cv2

from exfel_colormap import exfel_colormap_r as xfel
import numpy as np
import colorsys
import time
import matplotlib
from matplotlib.cm import ScalarMappable

import pdb

IMAGE_SIZE = [509,509,3]
WINDOW_OFFSET = [500,00]

def makeGaussian(size,sigma_x, sigma_y, pointing):
    if sigma_x == 0.0:
        sigma_x = 1e-6
    if sigma_y == 0.0:
        sigma_y = 1e-6
    if pointing == 0.0:
        pointing = 1e-6

    x, y = np.meshgrid(np.linspace(-np.round(size[1]/2),np.round(size[1]/2)-1,size[1]), np.linspace(-np.round(size[0]/2),np.round(size[0]/2)-1,size[0]))
    sigma_x *= size[0]/100.
    sigma_y *= size[0]/100.
    x0,y0 = np.random.normal(loc=0.0, scale=max(size)*pointing/1000., size=2)
    d2 = (x-x0)**2/sigma_x**2 + (y-y0)**2/sigma_y**2

    return np.exp(-d2 / 2. )

def nothing(arg):
    pass

def show_webcam():

    cam = VideoCapture(0)

    cv2.namedWindow('Control Room')
    cv2.namedWindow('Beam')
    cv2.namedWindow('Object')
    cv2.namedWindow('Illumination')
    cv2.namedWindow('Detector image')

    cv2.resizeWindow('Beam', IMAGE_SIZE[0], IMAGE_SIZE[1])
    cv2.resizeWindow('Object', IMAGE_SIZE[0], IMAGE_SIZE[1])
    cv2.resizeWindow('Illumination', IMAGE_SIZE[0], IMAGE_SIZE[1])
    cv2.resizeWindow('Detector image', IMAGE_SIZE[0], IMAGE_SIZE[1])

    cv2.moveWindow('Control Room', 0, 0 )
    cv2.moveWindow('Beam', WINDOW_OFFSET[0], WINDOW_OFFSET[1])
    cv2.moveWindow('Object', WINDOW_OFFSET[0]+IMAGE_SIZE[0], WINDOW_OFFSET[1])
    cv2.moveWindow('Illumination', WINDOW_OFFSET[0] + IMAGE_SIZE[0]//2, WINDOW_OFFSET[1] + IMAGE_SIZE[1])
    cv2.moveWindow('Detector image', WINDOW_OFFSET[0] + 3*IMAGE_SIZE[0]//2, WINDOW_OFFSET[1] + IMAGE_SIZE[1])

    cv2.createTrackbar('Beam Shutter', 'Control Room', 0, 1, nothing)
    cv2.createTrackbar('Horizontal beam diameter','Control Room',100,100,nothing)
    cv2.createTrackbar('Vertical beam diameter','Control Room',100,100,nothing)
    cv2.createTrackbar('Pointing stability', 'Control Room', 0, 100,nothing)
    cv2.createTrackbar('Open Camera', 'Control Room', 0, 1, nothing)

    while True:

        #pdb.set_trace()

        hor_beam_diameter = cv2.getTrackbarPos('Horizontal beam diameter','Control Room')
        ver_beam_diameter = cv2.getTrackbarPos('Vertical beam diameter','Control Room')
        pointing          = cv2.getTrackbarPos('Pointing stability','Control Room')
        camera_apt        = cv2.getTrackbarPos('Open Camera', 'Control Room')
        shutter           = cv2.getTrackbarPos('Beam Shutter', 'Control Room')


        img = np.zeros(shape=IMAGE_SIZE)
        if camera_apt > 0:
            img = imread("bean.jpeg",1)
            ret_val, img = cam.read()

        # Mirror
        img = flip(img, 1)

        greyimg = img.sum(axis=2)
        hsvimg = matplotlib.colors.rgb_to_hsv(img/255)

        hsv_value = hsvimg[:,:,2]

        beam = np.zeros(greyimg.shape)
        if shutter > 0:
            beam = makeGaussian(beam.shape, hor_beam_diameter, ver_beam_diameter, pointing)


        illumination = np.multiply(beam,greyimg)
        imgFT = np.log10(1.+np.abs(np.fft.fftshift(np.fft.fft2(np.fft.fftshift(illumination)))))

        imgFT -= imgFT.min()
        if imgFT.max() > 0:
            imgFT /= imgFT.max()

        ## For correct display images have to be normalized
        ## Add some background.
        ##imgFT = 1.0*imgFT + np.random.random(size=imgFT.shape) * background/1000. * imgFT.max()

        #if imgFT.max() > 0:
            #imgFT = imgFT/np.max(imgFT)
        #pdb.set_trace()

        beam_cmap_interface = ScalarMappable(norm=matplotlib.colors.Normalize(vmin=0.0, vmax=1.0), cmap='jet_r')
        viridis_beam = beam_cmap_interface.to_rgba(beam)

        imgFT_cmap_interface = ScalarMappable(norm=matplotlib.colors.Normalize(vmin=0.,vmax=1.0), cmap='jet_r')
        xfel_imgFT = imgFT_cmap_interface.to_rgba(imgFT)


        imshow('Beam',viridis_beam)
        imshow('Object',img)

        hsv_value *= beam
        hsvimg[:,:,2] = hsv_value
        illuminated_img = matplotlib.colors.hsv_to_rgb(hsvimg)

        imshow('Illumination',illuminated_img)
        imshow('Detector image',xfel_imgFT)

        #time.sleep(2)

        if waitKey(1) == 27:
            break  # esc to quit
    destroyAllWindows()

def main():
    show_webcam()

if __name__ == '__main__':
    main()

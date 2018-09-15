'''
Simply display the contents of the webcam with optional mirroring using OpenCV
via the new Pythonic cv2 interface.  Press <esc> to quit.
'''

import cv2
from cv2 import VideoCapture
from cv2 import flip
from cv2 import imread, imshow, imwrite
from cv2 import waitKey
from cv2 import destroyAllWindows

from exfel_colormap import exfel_colormap as xfel
import numpy as np
import colorsys
import time
import subprocess
import matplotlib
from matplotlib.cm import ScalarMappable

import pdb

IMAGE_SIZE = [640,480,3]
WINDOW_OFFSET = [320,0]

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

def show_webcam(camera_id=0):

    cv2.namedWindow('Kontrollraum')
    cv2.namedWindow('XFEL Strahl')
    cv2.namedWindow('Messobjekt')
    cv2.namedWindow('Bestrahlung')
    cv2.namedWindow('Streubild')

    cv2.resizeWindow('XFEL Strahl', IMAGE_SIZE[0], IMAGE_SIZE[1])
    cv2.resizeWindow('Messobjekt', IMAGE_SIZE[0], IMAGE_SIZE[1])
    cv2.resizeWindow('Bestrahlung', IMAGE_SIZE[0], IMAGE_SIZE[1])
    cv2.resizeWindow('Streubild', IMAGE_SIZE[0], IMAGE_SIZE[1])

    cv2.moveWindow('Kontrollraum', 0, 0 )
    cv2.moveWindow('XFEL Strahl', WINDOW_OFFSET[0], WINDOW_OFFSET[1])
    cv2.moveWindow('Messobjekt', WINDOW_OFFSET[0]+IMAGE_SIZE[0], WINDOW_OFFSET[1])
    cv2.moveWindow('Bestrahlung', WINDOW_OFFSET[0], WINDOW_OFFSET[1] + IMAGE_SIZE[1])
    cv2.moveWindow('Streubild', WINDOW_OFFSET[0] + IMAGE_SIZE[0], WINDOW_OFFSET[1] + IMAGE_SIZE[1])

    cv2.createTrackbar('XFEL offen', 'Kontrollraum', 1, 1, nothing)
    cv2.createTrackbar('Strahldurchmesser (hor.)','Kontrollraum',100,100,nothing)
    cv2.createTrackbar('Strahldurchmesser (ver.)','Kontrollraum',100,100,nothing)
    cv2.createTrackbar('Strahlstabilitaet', 'Kontrollraum', 0, 100,nothing)
    cv2.createTrackbar('Detektor offen', 'Kontrollraum', 1, 1, nothing)

    cam = VideoCapture(camera_id)

    while True:

        hor_beam_diameter = cv2.getTrackbarPos('Strahldurchmesser (hor.)','Kontrollraum')
        ver_beam_diameter = cv2.getTrackbarPos('Strahldurchmesser (ver.)','Kontrollraum')
        pointing          = cv2.getTrackbarPos('Strahlstabilitaet','Kontrollraum')
        camera_apt        = cv2.getTrackbarPos('Detektor offen', 'Kontrollraum')
        shutter           = cv2.getTrackbarPos('XFEL offen', 'Kontrollraum')


        if camera_apt > 0:
            #img = imread("bean.jpeg",1)
            ret_val, img = cam.read()

        # Mirror
        #img = flip(img, 1)

        greyimg = img.sum(axis=2)
        hsvimg = matplotlib.colors.rgb_to_hsv(img/255)

        hsv_value = hsvimg[:,:,2]

        beam = np.zeros(greyimg.shape)+1.e-6
        if shutter > 0:
            beam = makeGaussian(beam.shape, hor_beam_diameter, ver_beam_diameter, pointing)


        illumination = np.multiply(beam,greyimg)

        #imgFT = np.log10(1.+np.abs(np.fft.fftshift(np.fft.fft2(np.fft.fftshift(illumination)))))
        #imgFT -= imgFT.min()
        #if imgFT.max() > 0:
            #imgFT /= imgFT.max()

        imgFT = np.abs(np.fft.fftshift(np.fft.fft2(np.fft.fftshift(illumination))))

        colormap_beam = 'viridis'
        colormap_img = xfel
        beam_cmap_interface = ScalarMappable(norm=matplotlib.colors.Normalize(), cmap=colormap_beam)
        viridis_beam = rgba_to_bgra(beam_cmap_interface.to_rgba(beam, bytes=True))
        mn = imgFT.min()
        mx = imgFT.max()

        imgFT_cmap_interface = ScalarMappable(norm=matplotlib.colors.LogNorm(vmin=mn,vmax=mx), cmap=colormap_img)

        # Convert to RGBA (uint8 based).
        xfel_imgFT = rgba_to_bgra(imgFT_cmap_interface.to_rgba(imgFT, bytes=True))
        # Swap rgba to bgra (because different convensions between matplotlib and cv)


        imshow('XFEL Strahl',viridis_beam)
        imshow('Messobjekt',img)

        hsv_value *= beam
        hsvimg[:,:,2] = hsv_value
        illuminated_img = matplotlib.colors.hsv_to_rgb(hsvimg)

        imshow('Bestrahlung',illuminated_img)
        imshow('Streubild',xfel_imgFT)


        #time.sleep(2)
        k = waitKey(1)
        if k == 27:
            break  # esc to quit
        elif k == 65377:
            np.save('imgFT.npz', imgFT)
            take_snapshot(data=(img, xfel_imgFT))

    destroyAllWindows()

def take_snapshot(data):
    # Stitch images vertically.
    img1 = data[0]
    img2 = data[1][:,:,:-1]

    cv2.imwrite("/tmp/img1.jpg", img1)
    cv2.imwrite("/tmp/img2.jpg", img2)

    command = "convert /tmp/img1.jpg /tmp/img2.jpg -append /tmp/image.jpg && display /tmp/image.jpg"

    proc = subprocess.Popen(command, shell=True)

    k = waitKey()
    if k == 27:
        return  # esc to quit
    elif k == 65377:
        hcopy_cmd = "./print.sh /tmp/image.jpg"
        hcopy_proc = subprocess.Popen(hcopy_cmd, shell=True)

def rgba_to_bgra(img):
    r,g,b,a = cv2.split(img)
    return cv2.merge((b,g,r,a))

def main():
    show_webcam(camera_id=1)

if __name__ == '__main__':
    main()

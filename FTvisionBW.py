'''
Simply display the contents of the webcam with optional mirroring using OpenCV
via the new Pythonic cv2 interface.  Press <esc> to quit.
'''
from cv2 import VideoCapture
from cv2 import flip
from cv2 import imread, imshow
from cv2 import waitKey
from cv2 import destroyAllWindows

import numpy as np

def makeGaussian(size,sigma):
    x, y = np.meshgrid(np.linspace(-np.round(size[1]/2),np.round(size[1]/2)-1,size[1]), np.linspace(-np.round(size[0]/2),np.round(size[0]/2)-1,size[0]))
    d = np.sqrt(x*x+y*y)
    mu = 0.0
    return np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )

def show_webcam(mirror=False, noise=0):
    cam = VideoCapture(0)
    while True:

        # ret_val, img = cam.read()
        #ret_val, img = 0, np.random.random((1024,1024,3))
        #ret_val, img = 0
        img = imread("bean.jpeg",0)

        if mirror:
            img = np.sum(flip(img, 1),2)
            beam = makeGaussian(np.shape(img),50)

            imgFT = np.log10(np.abs(np.fft.fftshift(np.fft.fft2(np.fft.fftshift(np.multiply(beam,img))))))

            # For correct display images have to be normalized
            imgFT = imgFT/np.max(imgFT)
            img = img/np.max(img)

            if noise > 0.0:
                img += np.random.random(size=img.shape) * noise
                img = img/np.max(img)

            imshow('Real Space',np.multiply(img,beam))
            imshow('Reciprocal Space',imgFT)

        if waitKey(1) == 27:
            break  # esc to quit
    destroyAllWindows()

def main():
    show_webcam(mirror=True)

if __name__ == '__main__':
    main()

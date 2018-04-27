#! /usr/bin/env python
import matplotlib
from matplotlib.cm import ScalarMappable
#from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import sys
import numpy

import cv2
from cv2 import imread, imshow, imwrite, namedWindow, destroyAllWindows, waitKey

# Define colors
#hex_colors = ["#dadada", # light grey
              #"#b2b2b2", # darker grey
              #"#81b0c8", # 80% blue
              #"#a4c3d6", # 60% blue
              #"#c5d6e4", # 40% blue
              #"#e3ebf2", # 20% blue
              #"#f39200", # orange
              #"#559dbb", # blue
              #"#0d1546", # night blue
              #]

hex_colors = [
              "#e3ebf2", # 20% blue
              "#a4c3d6", # 60% blue
              "#0d1546", # night blue
              #"#513000", # 1/3 of orange
              "#794900", # 1/2 of orange
              #"#a26100", # 2/3 of orange
              "#f39200", # orange
              "#ffff00", # full orange
              "#ffffff", # white
]

hex_colors_r = [
              "#f39200", # orange
              "#a26100", # 2/3 of orange
              "#0d1546", # night blue
              "#a4c3d6", # 60% blue
              "#e3ebf2", # 20% blue
              "#ffffff", # white
]


# Fewer bins will result in "coarser" colomap interpolation
nbins = 256

# Setup the colormap by linear segmentation.
exfel_colormap = LinearSegmentedColormap.from_list( "exfel_colormap", hex_colors, N=nbins)
exfel_colormap_r = LinearSegmentedColormap.from_list( "exfel_colormap_r", hex_colors_r, N=nbins)


def main(args):

    if len(args) == 0:
        x,y = numpy.meshgrid(numpy.linspace(-4,4,81), numpy.linspace(-4,4,81))
        fake_data = 8*numpy.exp(-(x**2+y**2))
        plt.imshow(fake_data, cmap=exfel_colormap)

    elif len(args) == 1:
        fake_data = numpy.load(args[0])
        imgFT_cmap_interface = ScalarMappable(norm=matplotlib.colors.LogNorm(), cmap=exfel_colormap)

        # Convert to RGBA (uint8 based).
        xfel_imgFT = imgFT_cmap_interface.to_rgba(fake_data, bytes=True)

        # Swap rgba to bgra (because different convensions between matplotlib and cv)
        cv2_r, cv2_g, cv2_b, cv2_a = cv2.split(xfel_imgFT)
        xfel_imgFT = cv2.merge((cv2_b, cv2_g, cv2_r, cv2_a))
        namedWindow('Image')
        while True:
            imshow('Image', xfel_imgFT)
            k = waitKey(500)
            if k == 27:
                break  # esc to quit

        destroyAllWindows()

if __name__ == "__main__":
    main(sys.argv[1:])

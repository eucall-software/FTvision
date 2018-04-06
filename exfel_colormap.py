import matplotlib
import numpy

from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

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
              "#ffffff", # white
              "#e3ebf2", # 20% blue
              "#a4c3d6", # 60% blue
              "#0d1546", # night blue
              "#a26100", # 2/3 of orange
              "#f39200", # orange
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
nbins = 128

# Setup the colormap by linear segmentation.
exfel_colormap = LinearSegmentedColormap.from_list( "exfel_colormap", hex_colors, N=nbins)
exfel_colormap_r = LinearSegmentedColormap.from_list( "exfel_colormap_r", hex_colors_r, N=nbins)


def main():
    x,y = numpy.meshgrid(numpy.linspace(-4,4,81), numpy.linspace(-4,4,81))
    fake_data = 8*numpy.exp(-(x**2+y**2))
    plt.imshow(fake_data, cmap=exfel_colormap)
    plt.show()

if __name__ == "__main__":
    main()

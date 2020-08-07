# Code that loads all images made in R with the regression model and makes them use the afmhot colour map

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import glob

# Setting the font size for all plots.
matplotlib.rcParams['font.size'] = 9

# Change the file path here
file_path = r'thirdlayer/set2rm/**_R.png'  #Replace 2s for 4s as and when needed
# Change where the files are saved here
sav_dir = r'thirdlayer/set2rm/'

matches = glob.glob(file_path, recursive=True)
for file_path in matches:
    name = file_path.replace('thirdlayer/set2rm\\', '')
    name = name.replace('_R.png', '')
    sav_loc = sav_dir + name + '.png'

    image = matplotlib.image.imread(file_path)
    plt.imsave(sav_loc, image, cmap='afmhot')
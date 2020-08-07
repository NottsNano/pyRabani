import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import glob
from scipy import ndimage
import pandas as pd

# file_path = r'thirdlayer/set1res_good/21_2nd.png'
file_path = r'thirdlayer/set4perc_good_80/**_2nd.png'
sav_loc = r'thirdlayer/set4perc_good_80/'

matches = glob.glob(file_path, recursive=True)
for file_path in matches:
        name = file_path.replace('thirdlayer/set4perc_good_80\\', '')
        csv_sav_loc = file_path.replace('_2nd.png', '.csv')

        image = matplotlib.image.imread(file_path)

        image = ndimage.binary_closing(image[:, :, 0])
        image = ndimage.binary_opening(image)

        # plt.imshow(image)

        # plt.ion()

        lbl = ndimage.label(image)[0]
        com = ndimage.measurements.center_of_mass(image, lbl, list(range(1, np.max(
                lbl) + 1)))  # Fix this, make it into a list of two sets of x and y coordinates with zip and split
        cy, cx = list(zip(*com))

        area = np.zeros(np.max(lbl))

        for k in list(range(1, np.max(lbl) + 1)):
                A = np.count_nonzero(lbl == k)
                area[k - 1] = A

        dictio = {'Particle No.': list(range(1, np.max(lbl) + 1)),
                  'Area': list(area),
                  'X Centre': list(cx),
                  'Y Centre': list(cy)}

        df = pd.DataFrame(dictio)

        df.to_csv(csv_sav_loc, index=False)

        plt.imshow(image)
        plt.scatter(cx, cy, marker='+', s=1)
        plt.title(name)

        plt.show()



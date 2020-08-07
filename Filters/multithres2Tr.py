# Adapt the skimage example
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import glob
from skimage.filters import threshold_multiotsu

# Setting the font size for all plots.
matplotlib.rcParams['font.size'] = 9

# Change the file path here
file_path = r'thirdlayer/set3/**.png'
# Change where the files are saved here
sav_dir = r'thirdlayer/'

# Save a set of
good = ['6', '7', '8', '11', '17', '18', '19', '40', '57', '58', '59', '60', '61', '62', '63', '106', '110',
            '122', '133', '134', '141', '162', '170', '171', '173', '188', '201', '203', '205', '207',
            '209', '210', '217', '218', '244', '356', '371', '377', '378', '385', '450', '451', '472']  #47 / 99 are good!

# Make a loop that scans the directory
matches = glob.glob(file_path, recursive=True)
for file_path in matches:
    name = file_path.replace('thirdlayer/set3\\', '')
    name = name.replace('.png', '')
    sav_loc = sav_dir + 'set3res/' + name + 'tr_fig.png'
    pres_sav_loc = sav_dir + 'set3res/' + name + 'tr_res.png'  # Save regions images separately
    image = matplotlib.image.imread(file_path)
    # Applying multi-Otsu threshold for the default value, generating
    # three classes.
    thresholds = threshold_multiotsu(image)

    # Using the threshold values, we generate the three regions.
    regions = np.digitize(image, bins=thresholds)

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 3.5))

    # Plotting the original image.
    ax[0].imshow(image, cmap='gray')
    ax[0].set_title('Original')
    ax[0].axis('off')

    # Plotting the histogram and the two thresholds obtained from
    # multi-Otsu.
    ax[1].hist(image.ravel(), bins=255)
    ax[1].set_title('Histogram')
    for thresh in thresholds:
        ax[1].axvline(thresh, color='r')

    # Plotting the Multi Otsu result.
    ax[2].imshow(regions, cmap='afmhot')
    ax[2].set_title('Multi-Otsu result')
    ax[2].axis('off')

    plt.subplots_adjust()

    # plt.show()

    #Save the resulting figure and segmented image.
    plt.savefig(sav_loc)
    plt.imsave(pres_sav_loc, regions, cmap='afmhot')

    if name in good:
        gd_all_sav_loc = sav_dir + 'set3res_good/' + name + 'tr_all.png'
        plt.imsave(gd_all_sav_loc, regions, cmap='afmhot')

        gd_1st_sav_loc = sav_dir + 'set3res_good/' + name + 'tr_1st.png'
        plt.imsave(gd_1st_sav_loc, image > thresholds[0], cmap='afmhot')

        gd_2nd_sav_loc = sav_dir + 'set3res_good/' + name + 'tr_2nd.png'
        plt.imsave(gd_2nd_sav_loc, image > thresholds[1], cmap='afmhot')
    else:
        set4_sav_loc = sav_dir + 'set4/' + name + '.png'
        # plt.imsave(set4_sav_loc, image, cmap='gray')

        set4_all_sav_loc = sav_dir + 'set4m_o/' + name + 'tr_multiotsu.png'
        plt.imsave(set4_all_sav_loc, regions, cmap='afmhot')

    plt.close(fig)


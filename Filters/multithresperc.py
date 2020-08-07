# Adapt the skimage example but replace the 2nd threshold in multi-otsu with 0.8 or 0.9
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import glob
from skimage.filters import threshold_multiotsu

# Setting the font size for all plots.
matplotlib.rcParams['font.size'] = 9

# Change the file path here
file_path = r'thirdlayer/set4/**.png'
# Change where the files are saved here
sav_dir = r'thirdlayer/'

# Save a set of

good_80_set2 = ['57', '66', '82', '94', '191', '224', '239', '243', '244', '254', '303']
good_90_set2 = ['7', '122', '145', '147', '178', '181', '150', '182', '195', '202', '231', '262', '269', '262', '275',
                '314', '315']
good_80_set4 = ['10', '38', '39', '46', '82', '85', '90', '91', '92', '94', '97', '152', '186', '246', '284', '288',
                '289', '452', '470']
good_90_set4 = ['12', '15', '53', '54', '80', '96', '101', '102', '117', '142', '154', '155', '156', '174', '178',
                '181', '190', '198', '199', '200', '204', '212', '213', '216', '255', '263', '324', '426', '448',
                '449', '471', '473', '474']

# Make a loop that scans the directory
matches = glob.glob(file_path, recursive=True)
for file_path in matches:
    name = file_path.replace('thirdlayer/set4\\', '')
    name = name.replace('.png', '')
    sav_loc = sav_dir + 'set4perc_res/' + name + '_90_fig.png'
    pres_sav_loc = sav_dir + 'set4perc_res/' + name + '_90_res.png'  # Save regions images separately
    image = matplotlib.image.imread(file_path)
    # Applying multi-Otsu threshold for the default value, generating
    # three classes.
    thresholds = threshold_multiotsu(image)

    thresholds[1] = 0.8  # Set every threshold to 0.8

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

    if name in good_90_set4:
        gd_all_sav_loc = sav_dir + 'set4perc_good_90/' + name + 'Tr_90_all.png'
        plt.imsave(gd_all_sav_loc, regions, cmap='afmhot')

        gd_1st_sav_loc = sav_dir + 'set4perc_good_90/' + name + 'Tr_90_1st.png'
        plt.imsave(gd_1st_sav_loc, image > thresholds[0], cmap='afmhot')

        gd_2nd_sav_loc = sav_dir + 'set4perc_good_90/' + name + 'Tr_90_2nd.png'
        plt.imsave(gd_2nd_sav_loc, image > thresholds[1], cmap='afmhot')
    else:
        print('Skipped ' + name)
        # set2_sav_loc = set2_all_sav_loc = sav_dir + 'set2/' + name + '.png'
        # plt.imsave(set2_sav_loc, image, cmap='gray')

        # set2_all_sav_loc = sav_dir + 'set2perc_res/' + name + '_80perc.png'
        # plt.imsave(set2_all_sav_loc, regions, cmap='afmhot')


    plt.close(fig)


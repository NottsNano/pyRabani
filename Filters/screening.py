import os

import h5py
import numpy as np
import pycroscopy as scope
from matplotlib import pyplot as plt
from scipy import stats, ndimage, signal
from skimage import measure
from tensorflow.python.keras.models import load_model

from CNN.CNN_prediction import predict_with_noise, ImageClassifier
from CNN.get_stats import all_preds_histogram, all_preds_percentage
from Rabani_Generator.plot_rabani import show_image


class FileFilter:
    def __init__(self):
        self._igor_translator = scope.io.translators.IgorIBWTranslator(max_mem_mb=1024)
        self.image_res = None
        self.normalised_euler = None
        self.fail_reasons = []
        self.CNN_classification = None
        self.euler_classification = None
        self.filepath = None
        self.with_euler = True
        self.cats = ['liquid', 'hole', 'cellular', 'labyrinth', 'island']

    def assess_file(self, filepath, model, plot=False, savedir=None):
        """Try and filter the file"""
        data = norm_data = phase = median_data = flattened_data = binarized_data_for_plotting = binarized_data = img_classifier = img_classifier_euler = None
        self.filepath = filepath

        h5_file = self._load_ibw_file(filepath)
        if not self.fail_reasons:
            data, phase = self._parse_ibw_file(h5_file)

        if not self.fail_reasons:
            norm_data = self._normalize_data(data)
            phase = self._normalize_data(phase)

        if not self.fail_reasons:
            median_data = self._median_align(norm_data)
            median_phase = self._median_align(phase)
            self._is_image_noisy(median_data)
            self._is_image_noisy(median_phase)

        if not self.fail_reasons:
            flattened_data = self._poly_plane_flatten(median_data)

        if not self.fail_reasons:
            flattened_data = self._normalize_data(flattened_data)
            binarized_data, binarized_data_for_plotting = self._binarise(flattened_data)

        if not self.fail_reasons:
            img_classifier = self._CNN_classify(binarized_data, model)
            if self.with_euler:
                img_classifier_euler = self._euler_classify(binarized_data)

        if plot or savedir:
            self._plot(data, median_data, flattened_data, binarized_data, binarized_data_for_plotting, img_classifier,
                       img_classifier_euler, savedir)
            if not plot:
                plt.close()

    def _plot(self, data=None, median_data=None, flattened_data=None,
              binarized_data=None, binarized_data_for_plotting=None, img_classifier=None, img_classifier_euler=None, savedir=None):

        fig, axs = plt.subplots(2, 4)
        fig.tight_layout(pad=3)

        if data is not None:
            axs[0, 0].imshow(data, cmap='RdGy')
            axs[0, 0].set_title('Original Image')
            axs[0, 0].axis("off")
        if median_data is not None:
            axs[0, 1].imshow(median_data, cmap='RdGy')
            axs[0, 1].set_title('Median Aligned')
            axs[0, 1].axis("off")
        if flattened_data is not None:
            axs[0, 2].imshow(flattened_data, extent=(0, self.image_res, 0, self.image_res), origin='lower', cmap='RdGy')
            axs[0, 2].set_title('Planar Flattened')
            axs[0, 2].axis("off")
        if binarized_data_for_plotting is not None:
            thres = binarized_data_for_plotting[0]
            pix = binarized_data_for_plotting[1]
            pix_gauss_grad = binarized_data_for_plotting[2]
            peaks = binarized_data_for_plotting[3]
            troughs = binarized_data_for_plotting[4]

            axs[1, 0].plot(thres, pix_gauss_grad)
            axs[1, 0].scatter(thres[peaks], pix_gauss_grad[peaks])
            axs[1, 0].scatter(thres[troughs], pix_gauss_grad[troughs], marker='x')
            axs[1, 0].set_xlabel('Threshold')
            axs[1, 0].set_ylabel('Pixels')
            axs[1, 0].set_title('Thresholding Levels')
        if binarized_data is not None:
            show_image(binarized_data, axis=axs[0, 3])
            axs[0, 3].set_title('Binarized')
        if img_classifier is not None:
            all_preds_histogram(img_classifier.cnn_preds, self.cats, axis=axs[1, 1])
            all_preds_percentage(img_classifier.cnn_preds, self.cats, axis=axs[1, 2])
            axs[1, 1].set_title('Network Predictions')
            axs[1, 2].set_title('Network Predictions')
        if img_classifier_euler is not None:
            all_preds_percentage(img_classifier_euler.euler_preds, self.cats + ["none"], axis=axs[1, 3])
            axs[1, 3].set_title('Euler Predictions')

        if savedir:
            filename = os.path.basename(self.filepath)[:-4]
            plt.savefig(f"{savedir}/{filename}.png")

    def _load_ibw_file(self, filepath):
        try:
            translated_file = self._igor_translator.translate(file_path=filepath, verbose=False)
            h5_file = h5py.File(translated_file, mode='r')
            return h5_file
        except:
            self.fail_reasons += ["Corrupt file"]

    def _parse_ibw_file(self, h5_file):
        arr_data = np.array(h5_file['Measurement_000/Channel_000/Raw_Data'])
        arr_phase = np.array(h5_file['Measurement_000/Channel_002/Raw_Data'])

        if int(np.sqrt(len(arr_data))) == np.sqrt(len(arr_data)):
            self.image_res = int(np.sqrt(len(arr_data)))
            arr_data_reshaped = np.reshape(arr_data, (self.image_res, self.image_res))
            arr_phase_reshaped = np.reshape(arr_phase, (self.image_res, self.image_res))
            h5_file.close()
            return arr_data_reshaped, arr_phase_reshaped

        else:
            self.fail_reasons += ["Incomplete scan"]

    def _normalize_data(self, arr):
        return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

    def _median_align(self, arr):
        for i in range(1, self.image_res):
            diff = arr[i - 1, :] - arr[i, :]
            bins = np.linspace(np.min(diff), np.max(diff), 1000)
            binned_indices = np.digitize(diff, bins, right=True)
            np.sort(binned_indices)
            median_index = np.median(binned_indices)
            arr[i, :] += bins[int(median_index)]

        return arr

    def _is_image_noisy(self, arr):
        # TODO: Do a polynomial line fit & use chi fit to determine if line is noise
        dud_rows = 0
        for i in range(self.image_res):
            # If 95% of the values in the row != the mode, assume dead scan line
            row_mode = stats.mode(arr[i, :])
            counter = np.count_nonzero((0.95 * row_mode[0] < arr[i, :]) < 1.05 * row_mode[0])

            dud_rows = dud_rows + counter > 0.95 * self.image_res + np.prod(arr[i, :] == np.sort(arr[i, :])) + np.prod(
                arr[i, :] == np.sort(arr[i, :])[::-1])

            dud_rows += (counter > (0.95 * self.image_res)) + sum(arr[i, :] == np.sort(arr[i, :])) > (
                    0.95 * self.image_res) + sum(arr[i, :] == np.sort(arr[i, :])[::-1]) > (0.95 * self.image_res)

        is_noisy = dud_rows / self.image_res > 0.05
        if is_noisy:
            self.fail_reasons += ["Noisy image"]

        return is_noisy

    def _plane_flatten(self, arr):
        # Find the approximate direction of the gradient of the plane
        hor_gradient = np.mean(arr[:, 0] - arr[:, -1]) / self.image_res
        ver_gradient = np.mean(arr[-1, :] - arr[0, :]) / self.image_res

        # The options for gradients of the plane
        hor_grad_array = np.linspace(0 * hor_gradient, 1.5 * hor_gradient, 10)
        ver_grad_array = np.linspace(-1.5 * ver_gradient, 1.5 * ver_gradient, 10)

        square_differences = np.zeros((10, 10))
        centroid = ndimage.measurements.center_of_mass(arr)
        if not (0 <= int(centroid[0]) <= self.image_res):
            self.fail_reasons += ["Failed to plane flatten"]
            return None
        else:
            centroid_mass = arr[int(centroid[0]), int(centroid[1])]
            test_line_x = np.ones([self.image_res, self.image_res]) * range(-int(centroid[0]),
                                                                            self.image_res - int(centroid[0]))
            test_line_y = np.ones([self.image_res, self.image_res]) * range(-int(centroid[1]),
                                                                            self.image_res - int(centroid[1]))
            # Flatten pixel-by-pixel
            for i in range(10):
                for j in range(10):
                    hor_array = test_line_x * - hor_grad_array[i]
                    ver_array = np.transpose(test_line_y * - ver_grad_array[j])
                    square_differences[i, j] = np.sum(np.square(arr - (hor_array + ver_array + centroid_mass)))

            best_indices = np.unravel_index(np.argmin(square_differences, axis=None), square_differences.shape)
            hor_gradient = hor_grad_array[best_indices[0]]
            ver_gradient = ver_grad_array[best_indices[1]]

            hor_array = test_line_x * - hor_gradient
            ver_array = np.transpose(test_line_y * - ver_gradient)
            background_plane = hor_array + ver_array + centroid_mass

            return arr - background_plane

    def _poly_plane_flatten(self, arr, n=5):
        horz_mean = np.mean(arr, axis=0)  # averages all the columns into a x direction array
        vert_mean = np.mean(arr, axis=1)  # averages all the rows into a y direction array

        line_array = np.arange(self.image_res)

        horz_fit = np.polyfit(line_array, horz_mean, n)
        vert_fit = np.polyfit(line_array, vert_mean, n)

        horz_polyval = -np.poly1d(horz_fit)
        vert_polyval = -np.poly1d(vert_fit)

        xv, yv = np.meshgrid(horz_polyval(line_array), vert_polyval(line_array))

        return arr + yv + xv

    def _binarise(self, arr, nbins=1000, gauss_sigma=10):
        threshes = np.linspace(0, 1, nbins)
        pix = np.zeros((nbins,))
        for i, t in enumerate(threshes):
            pix[i] = np.sum(arr < t)

        pix_gauss_grad = ndimage.gaussian_gradient_magnitude(pix, gauss_sigma)
        peaks, properties = signal.find_peaks(pix_gauss_grad, prominence=1)
        troughs, properties = signal.find_peaks(-pix_gauss_grad, prominence=1)

        if len(troughs) < 1:
            self.fail_reasons += ["Failed to binarize"]
            return None, None
        else:
            return arr > threshes[troughs[len(troughs) - 1]], (threshes, pix, pix_gauss_grad, peaks, troughs)

    def _get_normalised_euler(self, arr):
        region = (measure.regionprops((arr != 0) + 1)[0])
        self.normalised_euler = region["euler_number"] / np.sum(arr != 0)

    def _CNN_classify(self, arr, model):
        img_classifier = predict_with_noise(img=arr, model=model, perc_noise=0.05, perc_std=0.001)

        # For each class find the mean CNN_classification
        max_class = int(np.argmax(img_classifier.cnn_majority_preds))
        if np.max(img_classifier.cnn_majority_preds) < 0.8:
            self.fail_reasons += ["CNN not confident enough"]
        if np.all(np.std(img_classifier.cnn_preds, axis=0) > 0.1):
            self.fail_reasons += ["CNN distributions too broad"]
        # if not self.fail_reasons:
        self.CNN_classification = self.cats[max_class]

        return img_classifier

    def _euler_classify(self, arr):
        img_classifier = ImageClassifier(arr, model=None)
        img_classifier.network_img_size = 128
        img_classifier.wrap_image()
        img_classifier.euler_classify()

        max_class = int(np.argmax(img_classifier.euler_majority_preds))
        if np.argmax(np.sum(img_classifier.euler_preds, axis=0) == len(self.cats)):
            self.fail_reasons += ["Euler category not clear enough"]
        if np.sum(img_classifier.euler_preds, axis=0)[max_class] <= 0.9 * len(img_classifier.euler_preds):
            self.fail_reasons += ["Euler distributions too broad"]

        cats = self.cats + ["none"]
        self.euler_classification = cats[max_class]

        return img_classifier


if __name__ == '__main__':
    model = load_model("/home/mltest1/tmp/pycharm_project_883/Data/Trained_Networks/2020-03-30--18-10/model.h5")

    test_filter = FileFilter()
    test_filter.assess_file(
        "Images/Parsed Dewetting 2020 for ML/thres_img/tp/Si_d10_ring5_05mgmL_0003.ibw",
        model, plot=True)
    print(test_filter.fail_reasons)

    test_filter = FileFilter()
    test_filter.assess_file(
        "Images/Parsed Dewetting 2020 for ML/thres_img/tp/SiO2_d10th_ring5_05mgmL_0002.ibw",
        model, plot=True)
    print(test_filter.fail_reasons)

    test_filter = FileFilter()
    test_filter.assess_file(
        "Images/Parsed Dewetting 2020 for ML/thres_img/tp/OH_0002.ibw",
        model, plot=True)
    print(test_filter.fail_reasons)

    test_filter = FileFilter()
    test_filter.assess_file(
        "Images/Parsed Dewetting 2020 for ML/thres_img/tp/000TEST.ibw",
        model, plot=True)
    print(test_filter.fail_reasons)

    test_filter = FileFilter()
    test_filter.assess_file(
        "Images/Parsed Dewetting 2020 for ML/thres_img/tp/SiO2_d10th_ring5_05mgmL_0004.ibw",
        model, plot=True)
    print(test_filter.fail_reasons)

    test_filter = FileFilter()
    test_filter.assess_file(
        "Images/Parsed Dewetting 2020 for ML/thres_img/tp/SiO2_d10th_ring5_05mgmL_0005.ibw",
        model, plot=True)
    print(test_filter.fail_reasons)

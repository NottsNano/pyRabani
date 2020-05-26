import os

import h5py
import numpy as np
import pycroscopy as scope
from matplotlib import pyplot as plt
from scipy import stats, ndimage, signal
from skimage import measure
from tensorflow.python.keras.models import load_model
import pyUSID as usid
from CNN.CNN_prediction import ImageClassifier
from CNN.get_stats import all_preds_histogram, all_preds_percentage
from Rabani_Generator.plot_rabani import show_image


class FileFilter:
    def __init__(self):
        self._igor_translator = scope.io.translators.IgorIBWTranslator(max_mem_mb=1024)
        self.image_res = self.image_size = None
        self.image_classifier = None
        self.normalised_euler = None
        self.binarized_data = None
        self.fail_reasons = None
        self.CNN_classification = None
        self.euler_classification = None
        self.filepath = None
        self.cats = ['liquid', 'hole', 'cellular', 'labyrinth', 'island']

    def assess_file(self, filepath, cnn_model=None, denoising_model=None, assess_euler=True, plot=False, savedir=None):
        """Load, preprocess, classify and filter a single real image.

        Parameters
        ----------
        filepath : str
            Path linking to a .ibw file to assess
        cnn_model : object of type tensorflow.category_model
            Optional. Tensorflow category_model containing categories FileFilter.cats.
            If None (default), only preprocessing will take place
        denoising_model : None or object of type tensorflow.category_model
            Optional. Tensorflow category_model to perform denoising. Default None
        assess_euler : bool
            Optional. If the normalised Euler characteristic should be assessed or not. Default True
        plot : bool
            Optional. If we should plot the results of preprocessing/assessment. Default False
        savedir : None or str
            Optional. If a string, save the plot to the given directory. Default None

        Examples
        --------
        >>> filterer = FileFilter
        >>> filterer.assess_file()
        """

        data = norm_data = phase = median_data = flattened_data = binarized_data_for_plotting = binarized_data = img_classifier = img_classifier_euler = None
        self.filepath = filepath

        try:
            h5_file = self._load_ibw_file(filepath)

            if not self.fail_reasons:
                data, phase = self._parse_ibw_file(h5_file)

            if not self.fail_reasons:
                norm_data = self._normalize_data(data)

            if not self.fail_reasons:
                median_data = self._median_align(norm_data)
                self._is_image_noisy(median_data)

            if not self.fail_reasons:
                flattened_data = self._poly_plane_flatten(median_data)

            if not self.fail_reasons:
                flattened_data = self._normalize_data(flattened_data)
                self.binarized_data, binarized_data_for_plotting = self._binarise(flattened_data)

            if not self.fail_reasons:
                self._are_lines_properly_binarised(self.binarized_data)

            if not self.fail_reasons:
                if denoising_model:
                    assessment_arr = self._wrap_image_to_tensorflow(self.binarized_data, cnn_model.input_shape[1])
                    assessment_arr = self._denoise(assessment_arr, denoising_model)
                else:
                    assessment_arr = self.binarized_data

                if category_model:
                    self.image_classifier = ImageClassifier(assessment_arr, cnn_model)
                    self._CNN_classify()

                    if assess_euler:
                        self._euler_classify()
        except:
            self.image_classifier = None
            self._add_fail_reason("Unexpected error")
            return None

        if plot or savedir:
            self._plot(data, median_data, flattened_data, self.binarized_data, binarized_data_for_plotting, savedir)
            if not plot:
                plt.close()

    def _plot(self, data=None, median_data=None, flattened_data=None,
              binarized_data=None, binarized_data_for_plotting=None,
              savedir=None):

        fig, axs = plt.subplots(2, 4)
        fig.tight_layout(pad=3)
        fig.suptitle(f"{os.path.basename(self.filepath)} - {self.fail_reasons}", fontsize=5)

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
        if self.image_classifier is not None:
            all_preds_histogram(self.image_classifier.cnn_preds, self.cats, axis=axs[1, 1])
            all_preds_percentage(self.image_classifier.cnn_preds, self.cats, axis=axs[1, 2])
            axs[1, 1].set_title('Network Predictions')
            axs[1, 2].set_title('Network Predictions')
        if self.image_classifier is not None:
            all_preds_percentage(self.image_classifier.euler_preds, self.cats + ["none"], axis=axs[1, 3])
            axs[1, 3].set_title('Euler Predictions')

        if savedir:
            filename = os.path.basename(self.filepath)[:-4]
            if self.fail_reasons:
                plt.savefig(f"{savedir}/fail/{filename}.png", dpi=300)
            else:
                plt.savefig(f"{savedir}/{self.CNN_classification}/{filename}.png", dpi=300)

    def _add_fail_reason(self, fail_str):
        if not self.fail_reasons:
            self.fail_reasons = [fail_str]
        else:
            self.fail_reasons += [fail_str]

    def _load_ibw_file(self, filepath):
        try:
            translated_file = self._igor_translator.translate(file_path=filepath, verbose=False)
            h5_file = h5py.File(translated_file, mode='r')
            return h5_file
        except:
            self._add_fail_reason("Corrupt file")

    def _parse_ibw_file(self, h5_file):
        if 'Measurement_000/Channel_000/Raw_Data' in h5_file.keys():
            arr_data = np.array(h5_file['Measurement_000/Channel_000/Raw_Data'])
            arr_phase = np.array(h5_file['Measurement_000/Channel_002/Raw_Data'])
        else:
            # print(f"{h5_file}")
            self._add_fail_reason("Corrupt scan")
            return None

        self.image_size = h5_file["Measurement_000"]["Position_Values"][-1, -1]

        if int(np.sqrt(len(arr_data))) == np.sqrt(len(arr_data)):
            self.image_res = int(np.sqrt(len(arr_data)))
            arr_data_reshaped = np.reshape(arr_data, (self.image_res, self.image_res))
            arr_phase_reshaped = np.reshape(arr_phase, (self.image_res, self.image_res))
            h5_file.close()
            return arr_data_reshaped, arr_phase_reshaped
        else:
            self._add_fail_reason("Incomplete scan")

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
        # Do >5% of rows have >95% of pixels the same value?
        modes = stats.mode(arr, axis=1)
        is_row_solid = modes[1] >= 0.95 * self.image_res
        is_noisy_solid = (np.sum(is_row_solid) / self.image_res) >= 0.05

        # Do >5% of rows have mean not within 80% of the value of the mean of the image?
        is_row_outlier = np.logical_or((np.mean(arr) * 0.2) >= np.mean(arr, 1),
                                       np.mean(arr, 1) >= (np.mean(arr) * 1.8))

        is_noisy_outlier = (np.sum(is_row_outlier) / self.image_res) >= 0.05

        is_noisy = is_noisy_solid or is_noisy_outlier
        if is_noisy:
            self._add_fail_reason("Noisy image")

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
            self._add_fail_reason("Failed to plane flatten")
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
        peaks, properties = signal.find_peaks(pix_gauss_grad, prominence=50)
        troughs, properties = signal.find_peaks(-pix_gauss_grad, prominence=50)

        if len(troughs) < 1:
            self._add_fail_reason("Failed to binarise")
            return None, None
        else:
            return arr > threshes[troughs[len(troughs) - 1]], (threshes, pix, pix_gauss_grad, peaks, troughs)

    def _are_lines_properly_binarised(self, arr):
        unique_lines = np.unique(arr, axis=0)
        are_lines_improperly_binarised = len(unique_lines) < self.image_res - 20
        if are_lines_improperly_binarised:
            self._add_fail_reason("Corrupt binarisation")

        return not are_lines_improperly_binarised

    def _get_normalised_euler(self, arr):
        region = (measure.regionprops((arr != 0) + 1)[0])
        self.normalised_euler = region["euler_number"] / np.sum(arr != 0)

    @staticmethod
    def _wrap_image_to_tensorflow(img, network_img_size, jump_size=8):
        return ImageClassifier._wrap_image_to_tensorflow(img, network_img_size, jump_size)

    def _denoise(self, arr, denoising_model):
        return np.round(denoising_model.predict(arr))

    def _CNN_classify(self):
        self.image_classifier.cnn_classify()

        # For each class find the mean CNN_classification
        max_class = int(np.argmax(self.image_classifier.cnn_majority_preds))

        if np.max(self.image_classifier.cnn_majority_preds) < 0.8:
            self._add_fail_reason("CNN not confident enough")
        if np.any(np.std(self.image_classifier.cnn_preds, axis=0) > 0.1):
            self._add_fail_reason("CNN distributions too broad")
        if np.sum(self.image_classifier.cnn_preds[:, max_class] >= 0.9999) > (0.98 * len(
                self.image_classifier.cnn_preds)):
            self._add_fail_reason("CNN overfit")

        self.CNN_classification = self.cats[max_class]

    def _euler_classify(self):
        self.image_classifier.euler_classify()

        max_class = int(np.argmax(self.image_classifier.euler_majority_preds))
        if np.max(self.image_classifier.euler_majority_preds) < 0.8:
            self._add_fail_reason("Euler not confident enough")
        if np.any(np.std(self.image_classifier.euler_preds, axis=0) > 0.1):
            self._add_fail_reason("Euler distributions too broad")
        if np.argmax(np.sum(self.image_classifier.euler_preds, axis=0) == len(self.cats)):
            self._add_fail_reason("Euler cannot classify")

        cats = self.cats + ["none"]
        self.euler_classification = cats[max_class]


if __name__ == '__main__':
    category_model = load_model(
        "/home/mltest1/tmp/pycharm_project_883/Data/Trained_Networks/2020-03-30--18-10/model.h5")
    denoising_model = load_model(
        "/home/mltest1/tmp/pycharm_project_883/Data/Trained_Networks/2020-05-19--19-36/model.h5")

    test_filter = FileFilter()
    test_filter.assess_file(
        "Images/Parsed Dewetting 2020 for ML/thres_img/tp/Si_d10_ring5_05mgmL_0003.ibw",
        category_model, denoising_model, plot=True)
    print(test_filter.fail_reasons)

    test_filter = FileFilter()
    test_filter.assess_file(
        "Images/Parsed Dewetting 2020 for ML/thres_img/tp/SiO2_d10th_ring5_05mgmL_0002.ibw",
        category_model, denoising_model, plot=False)
    print(test_filter.fail_reasons)

    test_filter = FileFilter()
    test_filter.assess_file(
        "Images/Parsed Dewetting 2020 for ML/thres_img/tp/OH_0002.ibw",
        category_model, denoising_model, plot=False)
    print(test_filter.fail_reasons)

    test_filter = FileFilter()
    test_filter.assess_file(
        "Images/Parsed Dewetting 2020 for ML/thres_img/tp/000TEST.ibw",
        category_model, denoising_model, plot=False)
    print(test_filter.fail_reasons)

    test_filter = FileFilter()
    test_filter.assess_file(
        "Images/Parsed Dewetting 2020 for ML/thres_img/tp/SiO2_d10th_ring5_05mgmL_0004.ibw",
        category_model, denoising_model, plot=False)
    print(test_filter.fail_reasons)

    test_filter = FileFilter()
    test_filter.assess_file(
        "Images/Parsed Dewetting 2020 for ML/thres_img/tp/SiO2_d10th_ring5_05mgmL_0005.ibw",
        category_model, denoising_model, plot=False)
    print(test_filter.fail_reasons)

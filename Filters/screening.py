import h5py
import numpy as np
import pycroscopy as scope
from matplotlib import pyplot as plt
from scipy import stats, ndimage, signal
from tensorflow.python.keras.models import load_model

from CNN.CNN_prediction import predict_with_noise
from CNN.get_stats import all_preds_histogram
from Rabani_Generator.plot_rabani import show_image


class FileFilter:
    def __init__(self):
        self._igor_translator = scope.io.translators.IgorIBWTranslator(max_mem_mb=1024)
        self.image_res = None
        self.fail_reason = None
        self.classification = None
        self.cats = ['liquid', 'hole', 'cellular', 'labyrinth', 'island']

    def assess_file(self, filepath, model, plot=False):
        """Try and filter the file"""
        data = phase = median_data = flattened_data = binarized_data_for_plotting = binarized_data = img_classifier = None

        h5_file = self._load_ibw_file(filepath)
        if not self.fail_reason:
            data, phase = self._parse_ibw_file(h5_file)

        if not self.fail_reason:
            data = self._normalize_data(data)
            phase = self._normalize_data(phase)

        if not self.fail_reason:
            median_data = self._median_align(data)
            median_phase = self._median_align(phase)
            self._is_image_noisy(median_data)
            self._is_image_noisy(median_phase)

        if not self.fail_reason:
            flattened_data = self._plane_flatten(median_data)
            flattened_data = self._normalize_data(flattened_data)
            binarized_data, binarized_data_for_plotting = self._binarise(flattened_data)

        if not self.fail_reason:
            img_classifier = self._CNN_classify(binarized_data, model)

        if plot:
            self._plot(data, median_data, flattened_data, binarized_data, binarized_data_for_plotting, img_classifier)

    def _plot(self, data=None, median_data=None, flattened_data=None,
              binarized_data=None, binarized_data_for_plotting=None, img_classifier=None):

        fig, axs = plt.subplots(2, 3)
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
            show_image(binarized_data, axis=axs[1, 1])
            axs[1, 1].set_title('Binarized')
        if img_classifier is not None:
            all_preds_histogram(img_classifier.preds, self.cats, axis=axs[1, 2])
            axs[1, 2].set_title('Network Predictions')

    def _load_ibw_file(self, filepath):
        try:
            translated_file = self._igor_translator.translate(file_path=filepath, verbose=False)
            h5_file = h5py.File(translated_file, mode='r')
            return h5_file
        except:
            self.fail_reason = "Corrupt file or wrong file extension"

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
            self.fail_reason = "Incomplete scan"

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

        is_noisy = dud_rows / self.image_res > 0.05
        if is_noisy:
            self.fail_reason = "Noisy image"

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

    def _binarise(self, arr, nbins=1000, gauss_sigma=10):
        threshes = np.linspace(0, 1, nbins)
        pix = np.zeros((nbins,))
        for i, t in enumerate(threshes):
            pix[i] = np.sum(arr < t)

        pix_gauss_grad = ndimage.gaussian_gradient_magnitude(pix, gauss_sigma)
        peaks, properties = signal.find_peaks(pix_gauss_grad, prominence=1)
        troughs, properties = signal.find_peaks(-pix_gauss_grad, prominence=1)

        if len(troughs) not in [1, 2]:
            self.fail_reason = "Failed to binarize"
            return None, None
        else:
            return arr > threshes[troughs[0]], (threshes, pix, pix_gauss_grad, peaks, troughs)

    def _CNN_classify(self, arr, model):
        img_classifier = predict_with_noise(img=arr, model=model, perc_noise=0.05, perc_std=0.001)

        # For each class find the mean classification
        max_class = int(np.argmax(img_classifier.majority_preds))

        if np.max(img_classifier.majority_preds) < 0.8:
            self.fail_reason = "CNN not confident enough"
        elif np.all(np.std(img_classifier.preds, axis=0) > 0.1):
            self.fail_reason = "CNN distributions too broad"
        else:
            self.classification = self.cats[max_class]

        return img_classifier


if __name__ == '__main__':
    model = load_model("/home/mltest1/tmp/pycharm_project_883/Data/Trained_Networks/2020-03-30--18-10/model.h5")

    test_filter = FileFilter()
    test_filter.assess_file(
        "Images/Parsed Dewetting 2020 for ML/thres_img/tp/Si_d10_ring5_05mgmL_0003.ibw",
        model, plot=True)
    print(test_filter.fail_reason)

    test_filter = FileFilter()
    test_filter.assess_file(
        "Images/Parsed Dewetting 2020 for ML/thres_img/tp/SiO2_d10th_ring5_05mgmL_0002.ibw",
        model, plot=True)
    print(test_filter.fail_reason)

    test_filter = FileFilter()
    test_filter.assess_file(
        "Images/Parsed Dewetting 2020 for ML/thres_img/tp/OH_0002.ibw",
        model, plot=True)
    print(test_filter.fail_reason)

    test_filter = FileFilter()
    test_filter.assess_file(
        "Images/Parsed Dewetting 2020 for ML/thres_img/tp/000TEST.ibw",
        model, plot=True)
    print(test_filter.fail_reason)

    test_filter = FileFilter()
    test_filter.assess_file(
        "Images/Parsed Dewetting 2020 for ML/thres_img/tp/SiO2_d10th_ring5_05mgmL_0004.ibw",
        model, plot=True)
    print(test_filter.fail_reason)

    test_filter = FileFilter()
    test_filter.assess_file(
        "Images/Parsed Dewetting 2020 for ML/thres_img/tp/SiO2_d10th_ring5_05mgmL_0005.ibw",
        model, plot=True)
    print(test_filter.fail_reason)

from CNN.CNN_training import h5RabaniDataGenerator
from Filters.alignerwthreshold import tmp_img_loader
import numpy as np
import itertools

def validation_pred_generator(model, validation_datadir, y_params, y_cats, batch_size):
    validation_generator = h5RabaniDataGenerator(validation_datadir, batch_size=batch_size,
                                                 is_train=False, imsize=128)
    validation_generator.is_validation_set = True

    validation_preds = model.predict_generator(validation_generator, steps=validation_generator.__len__())
    validation_truth = validation_generator.y_true

    return validation_preds, validation_truth

def validation_pred_image(model, image):
    pass

def wrap_image(img, window_jump, final_img_size=128):
    # Figure out how many "windows" to make
    num_jumps = int((len(img) - final_img_size) / window_jump)
    jump_idx = itertools.product(np.arange(num_jumps), np.arange(num_jumps))

    # Copy each window out
    out_arr = np.zeros((num_jumps ** 2, final_img_size, final_img_size, 1))
    for i, (jump_i, jump_j) in enumerate(jump_idx):
        out_arr[i, :, :, 0] = img[(jump_i * window_jump): (jump_i * window_jump) + final_img_size,
                              (jump_j * window_jump): (jump_j * window_jump) + final_img_size]

    return out_arr

# Load an image
img = tmp_img_loader("Images/Parsed Dewetting 2020 for ML/thres_img/tp/000TEST.ibw")



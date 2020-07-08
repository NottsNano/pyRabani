import datetime
import os

from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.models import load_model

from Analysis.model_stats import plot_model_history
from Models.h5_iterator import h5RabaniDataGenerator
from Models.model_CNN import get_model, autoencoder


def train_CNN(model_dir, train_datadir, test_datadir, y_params, y_cats, batch_size, epochs, imsize, network_type):
    # Set up generators
    train_generator = h5RabaniDataGenerator(train_datadir, network_type=network_type, batch_size=batch_size,
                                            is_train=True, imsize=imsize,
                                            output_parameters_list=y_params, output_categories_list=y_cats)
    test_generator = h5RabaniDataGenerator(test_datadir, network_type=network_type, batch_size=batch_size,
                                           is_train=False, imsize=imsize,
                                           output_parameters_list=y_params, output_categories_list=y_cats)

    # Set up model
    if network_type == "classifier":
        model = get_model("VGG", (imsize, imsize, 1), len(y_cats), Adam())
    elif network_type == "autoencoder":
        model = autoencoder((imsize, imsize, 1), optimiser=Adam())
    else:
        raise ValueError("network_type must be one of ['classifier', 'autoencoder']")
    model = load_model("/home/mltest1/tmp/pycharm_project_883/Data/Trained_Networks/2020-05-29--10-48/model.h5")

    # early_stopping = EarlyStopping(monitor="val_loss", patience=10)
    model_checkpoint = ModelCheckpoint(get_model_storage_path(model_dir), monitor="val_loss", save_best_only=True)

    # Train
    model.fit_generator(generator=train_generator,
                        validation_data=test_generator,
                        steps_per_epoch=train_generator.__len__() // 10,
                        validation_steps=test_generator.__len__(),
                        class_weight=train_generator.class_weights_dict,
                        epochs=epochs,
                        max_queue_size=100)

    return model


def validate_CNN(model, validation_datadir, network_type, y_params, y_cats, batch_size, imsize=128,
                 steps=None):
    """Prediction generator for simulated validation data"""
    validation_generator = h5RabaniDataGenerator(validation_datadir, network_type=network_type, batch_size=batch_size,
                                                 is_train=False, imsize=imsize, output_parameters_list=y_params,
                                                 output_categories_list=y_cats, force_binarisation=True)
    validation_generator.is_validation_set = True

    if not steps:
        steps = validation_generator.__len__()

    if network_type == "classifier":
        validation_preds = model.predict_generator(validation_generator, steps=steps)[:steps * batch_size, :]
        validation_truth = validation_generator.y_true[:steps * batch_size, :]
    elif network_type == "autoencoder":
        validation_preds = model.predict_generator(validation_generator, steps=steps)[:steps * batch_size, :, :, :]
        validation_truth = validation_generator.x_true[:steps * batch_size, :, :, :]
    else:
        raise ValueError("Network type must be 'classifier' or 'autoencoder")

    return validation_preds, validation_truth


def get_model_storage_path(root_dir):
    current_datetime = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M")
    os.mkdir(f"{root_dir}/{current_datetime}")
    model_path = f"{root_dir}/{current_datetime}/model.h5"

    return model_path


def save_model(model, root_dir):
    model_path = get_model_storage_path(root_dir)
    model.save(model_path)


if __name__ == '__main__':
    # Train
    training_data_dir = "/home/mltest1/tmp/pycharm_project_883/Data/Simulated_Images/TrainFinal"  # "/media/mltest1/Dat Storage/pyRabani_Images"
    testing_data_dir = "/home/mltest1/tmp/pycharm_project_883/Data/Simulated_Images/NewTest"  # "/home/mltest1/tmp/pycharm_project_883/Images/2020-03-09/16-51"
    validation_data_dir = "/home/mltest1/tmp/pycharm_project_883/Data/Simulated_Images/2020-03-25/13-59"

    original_categories = ["liquid", "hole", "cellular", "labyrinth", "island"]
    original_parameters = ["kT", "mu"]

    trained_model = train_CNN(model_dir="Data/Trained_Networks", train_datadir=training_data_dir,
                              test_datadir=testing_data_dir,
                              y_params=original_parameters, y_cats=original_categories, batch_size=128,
                              imsize=200, epochs=8, network_type="classifier")

    # trained_model = train_autoencoder(model_dir="Data/Trained_Networks", train_datadir=training_data_dir,
    #                                   test_datadir=testing_data_dir,
    #                                   y_params=original_parameters, y_cats=original_categories, batch_size=128,
    #                                   imsize=200,
    #                                   epochs=15)

    plot_model_history(trained_model)
    # visualise_autoencoder_preds(trained_model, simulated_datadir=testing_data_dir,
    #                             good_datadir="/home/mltest1/tmp/pycharm_project_883/Data/Autoencoder_Testing/Good_Images",
    #                             bad_datadir="/home/mltest1/tmp/pycharm_project_883/Data/Autoencoder_Testing/Bad_Images")

    save_model(trained_model, "/home/mltest1/tmp/pycharm_project_883/Data/Trained_Networks")

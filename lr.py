import json
import os
import sys

import numpy as np
import tensorflow as tf
from numpy import ndarray as nda
from tensorflow import keras
from timeit import default_timer


def save_to_file(filename: str, what: dict):
    with open(filename, "w") as f:
        f.write(json.dumps(what, indent=4))


def apply_label_noise(labels: nda, noise: float) -> nda:
    noisy_labels = np.copy(labels)
    label_values = np.unique(noisy_labels)
    flip_indices = np.nonzero(np.random.binomial(n=1, p=noise, size=noisy_labels.size))[
        0]
    new_labels = np.random.choice(label_values, size=flip_indices.size).reshape(-1, 1)
    noisy_labels[flip_indices] = new_labels

    return noisy_labels


def train_model_cross_entropy(train_f: nda, train_l: nda, test_f: nda, test_l: nda,
                              classes: int, l2_reg: float,
                              sgd_lr: float) -> float:
    dimension = train_f.shape[1]
    model = keras.models.Sequential([
        keras.layers.Dense(classes, input_shape=(dimension,), activation='softmax',
                           kernel_regularizer=tf.keras.regularizers.l2(l2_reg))
    ])
    model.compile(optimizer=keras.optimizers.SGD(learning_rate=sgd_lr, momentum=0.9),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_f, train_l, epochs=200, batch_size=64,
              validation_data=(test_f, test_l), verbose=1)
    loss_accuracy = model.evaluate(test_f, test_l, verbose=1, return_dict=True)
    return 1 - loss_accuracy["accuracy"]


if __name__ == "__main__":
    # Copied from https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth
    # This prevents CUBLAS_STATUS_NOT_INITIALIZED masking error CUDA_ERROR_OUT_OF_MEMORY
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            pass

    # 1. Path to file(s)
    path = sys.argv[1]
    if path.endswith(".npz"):
        filenames = [path]
    else:
        filenames = list(
            filter(lambda x: ".npz" in x,
                   map(lambda x: os.path.join(path, x.name),
                       filter(lambda x: x.is_file(), os.scandir(path))))
        )
    print(f"Processing following files: {filenames}")

    # 2. result file
    result_filename = sys.argv[2]

    # 3. varying noise levels
    also_noise = sys.argv[3].lower() in {"y", "yes"}
    if also_noise:
        noise_levels = [None, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    else:
        noise_levels = [None]
    print(f"Using noise levels: {noise_levels}")

    # Compute everything
    result_json = {}
    for filename_ in filenames:
        dataset_name, embedding_name = filename_.split(".")[0].split("/")[-1].rsplit("-", 1)
        print(dataset_name, embedding_name)

        # Prepare dictionary
        if dataset_name not in result_json:
            result_json[dataset_name] = {}
        if embedding_name not in result_json[dataset_name]:
            result_json[dataset_name][embedding_name] = {}

        data = np.load(filename_)
        train_f_ = data["train_features"]
        train_l_ = data["train_labels"]
        test_f_ = data["test_features"]
        test_l_ = data["test_labels"]
        classes_ = np.unique(test_l_).size

        for noise_level in noise_levels:
            if noise_level is None:
                noise_level_str = "0.0"
            else:
                noise_level_str = str(noise_level)
            result_json[dataset_name][embedding_name][noise_level_str] = {}

            for l2_reg_ in [0.0, 0.0001, 0.001, 0.01, 0.1]:
                for sgd_lr_ in [0.0001, 0.001, 0.01, 0.1]:
                    errs = []
                    times = []
                    for i in range(5):
                        if noise_level is None:
                            noisy_train_l_ = train_l_
                            noisy_test_l_ = test_l_
                        else:
                            noisy_train_l_ = apply_label_noise(train_l_, noise_level)
                            noisy_test_l_ = apply_label_noise(test_l_, noise_level)

                        print(
                            f"Processing: {filename_}, noise: {noise_level_str}, L2: {l2_reg_}, "
                            f"SGD LR: {sgd_lr_}, rep: {i + 1}/5")
                        start = default_timer()
                        err = train_model_cross_entropy(train_f=train_f_,
                                                        train_l=noisy_train_l_,
                                                        test_f=test_f_,
                                                        test_l=noisy_test_l_,
                                                        classes=classes_,
                                                        l2_reg=l2_reg_,
                                                        sgd_lr=sgd_lr_)
                        time = default_timer() - start
                        errs.append(err)
                        times.append(time)
                    result_json[dataset_name][embedding_name][noise_level_str][
                        str((l2_reg_, sgd_lr_))] = \
                        {"errors": errs, "times": times}

                    save_to_file(result_filename, result_json)

import os
import sys
from typing import Dict, List

import numpy as np

# Values of label noise (besides noiseless case)
GLOBAL_NOISE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def _get_permutation_indices(size: int):
    return np.random.choice(np.arange(size), size, replace=False)


class EmbeddingToConvergence:
    def __init__(self, np_embedding_file_path: str, low_memory: bool = False, cosine_distance: bool = False):
        print(f"Loading file: {np_embedding_file_path}")
        np_obj = np.load(np_embedding_file_path)
        self._path = np_embedding_file_path

        self._train = np_obj["train_features"]
        self._test = np_obj["test_features"]
        self._train_labels = np_obj["train_labels"].reshape(-1)
        self._test_labels = np_obj["test_labels"].reshape(-1)
        self._num_train = self._train.shape[0]
        self._num_test = self._test.shape[0]
        self._dimension = self._train.shape[1]
        self._cosine_distance = cosine_distance

        if cosine_distance:
            print("Performing normalization")
            self._train = self._train / np.linalg.norm(self._train, axis=1).reshape(-1, 1)
            self._test = self._test / np.linalg.norm(self._test, axis=1).reshape(-1, 1)
            print("Done performing normalization")

        print(f"Done loading file: {np_embedding_file_path}")

        self._low_memory = low_memory
        if not low_memory:
            print("Computing global distance matrix")
            self._distance_matrix = self._compute_distance_matrix(self._train, self._test)
            print("Done computing global distance matrix")

    @staticmethod
    def _compute_distance_matrix(train: np.ndarray, test: np.ndarray) -> np.ndarray:
        x_xt = np.matmul(test, np.transpose(train))
        x_train_2 = np.sum(np.square(train), axis=1)
        x_test_2 = np.sum(np.square(test), axis=1)

        for i in range(np.shape(x_xt)[0]):
            x_xt[i, :] = np.multiply(x_xt[i, :], -2)
            x_xt[i, :] = np.add(x_xt[i, :], x_test_2[i])
            x_xt[i, :] = np.add(x_xt[i, :], x_train_2)

        return x_xt

    def _permute_train(self):
        permutation_indices = _get_permutation_indices(self._num_train)
        self._train = self._train[permutation_indices, :]
        self._train_labels = self._train_labels[permutation_indices]

        if not self._low_memory:
            self._distance_matrix = self._distance_matrix[:, permutation_indices]

    @staticmethod
    def _apply_label_noise(labels: np.ndarray, noise_level: float) -> np.ndarray:
        return_labels = np.copy(labels)
        label_values = np.unique(return_labels)

        # Pick samples for which labels will be flipped
        flip_indices = np.nonzero(np.random.binomial(n=1, p=noise_level, size=return_labels.size))[0]
        new_labels = np.random.choice(label_values, size=flip_indices.size)

        return_labels[flip_indices] = new_labels
        return return_labels

    def _simulate_convergence_once(self, noise_levels: List[float]) -> Dict[float, np.ndarray]:
        self._permute_train()

        errs = {}
        train_labels_flipped = {}
        test_labels_flipped = {}

        for noise_level in noise_levels:
            errs[noise_level] = np.zeros((self._num_train + 1,), dtype=np.int)
            errs[noise_level][0] = self._num_test

            # Apply label flipping
            train_labels_flipped[noise_level] = self._apply_label_noise(self._train_labels, noise_level=noise_level)
            test_labels_flipped[noise_level] = self._apply_label_noise(self._test_labels, noise_level=noise_level)

        min_distances = np.full((self._num_test,), fill_value=np.Inf, dtype=np.float)
        min_index = np.zeros((self._num_test,), dtype=np.int)

        # Used to store subset of the global distance matrix
        subset_matrix = None
        subset_size = 10_000

        for index_of_new_train_point in range(self._num_train):
            # Use precomputed distance matrix
            if not self._low_memory:
                current_distances = self._distance_matrix[:, index_of_new_train_point].reshape(-1)

            # Compute relevant part of the distance matrix at a time
            else:
                # Compute a smaller subset of global distance matrix
                if index_of_new_train_point % subset_size == 0:
                    start_index = index_of_new_train_point
                    end_index = index_of_new_train_point + subset_size
                    subset_matrix = self._compute_distance_matrix(self._train[start_index:end_index, :], self._test)

                if index_of_new_train_point % (5 * subset_size) == 0:
                    print(f"\tProcessed {index_of_new_train_point}/{self._num_train} training points")

                current_distances = subset_matrix[:, index_of_new_train_point % subset_size].reshape(-1)

            test_indices_improved = current_distances < min_distances
            min_distances[test_indices_improved] = current_distances[test_indices_improved]
            min_index[test_indices_improved] = index_of_new_train_point

            for noise_level in noise_levels:
                predicted_labels = train_labels_flipped[noise_level][min_index]
                errs[noise_level][index_of_new_train_point + 1] = \
                    np.count_nonzero(predicted_labels != test_labels_flipped[noise_level])

        return errs

    def simulate_convergence(self, also_noise: bool, num_reps: int):
        noise_levels = [0.0]
        if also_noise:
            global GLOBAL_NOISE_LEVELS
            noise_levels.extend(GLOBAL_NOISE_LEVELS)

        # #reps x errors when using 0 -> all train points
        result = {}
        for noise_level in noise_levels:
            result[noise_level] = np.empty((num_reps, self._num_train + 1), dtype=np.int)

        for i in range(num_reps):
            print(f"Repetition: {i + 1}/{num_reps} for embeddings at: {self._path}")
            errs = self._simulate_convergence_once(noise_levels=noise_levels)

            for noise_level in noise_levels:
                result[noise_level][i, :] = errs[noise_level]

        store_path = self._path.split(".")[0] + "-errs"
        if self._cosine_distance:
            store_path += "-cosine"

        for noise_level in noise_levels:
            store_path_current_noise_level = store_path + "-" + str(noise_level)
            np.save(store_path_current_noise_level, result[noise_level])


if __name__ == "__main__":
    path = "results/"
    dataset_name = sys.argv[1]

    assert dataset_name in {"cifar10", "cifar100", "mnist", "yelp", "sst2", "imdb_reviews"}
    if dataset_name in {"cifar10", "cifar100", "mnist"}:
        embed_names = ['alexnet', 'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3',
                       'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7',
                       'googlenet', 'inception', 'resnet_101_v2', 'resnet_152_v2', 'resnet_50_v2', 'vgg16',
                       'vgg19', 'pca_32', 'pca_64', 'pca_128', 'raw']

    else:
        embed_names = ['bert_cased', 'bert_cased_pool', 'bert_uncased_large_pool', 'nnlm_128',
                       'nnlm_50_normalization', 'xlnet', 'bert_cased_large', 'bert_uncased',
                       'bert_uncased_pool', 'nnlm_128_normalization', 'use', 'xlnet_large',
                       'bert_cased_large_pool', 'bert_uncased_large', 'elmo', 'nnlm_50', 'use_large']

    for embed_name in embed_names:
        etc = EmbeddingToConvergence(os.path.join(path, f"{dataset_name}-{embed_name}.npz"),
                                     low_memory=dataset_name == "yelp", cosine_distance=True)
        etc.simulate_convergence(also_noise=False, num_reps=10 if dataset_name == "yelp" else 30)

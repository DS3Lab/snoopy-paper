import json
import math
import os
import sys
from typing import List, Tuple

import numpy as np
from disjoint_set import DisjointSet
from numpy import ndarray as nd


def distance_matrix_reduced_numpy(train: nd, test: nd) -> nd:
    x_xt = np.matmul(test, np.transpose(train))
    x_train_2 = np.sum(np.square(train), axis=1)
    x_test_2 = np.sum(np.square(test), axis=1)

    for i in range(np.shape(x_xt)[0]):
        x_xt[i, :] = np.multiply(x_xt[i, :], -2)
        x_xt[i, :] = np.add(x_xt[i, :], x_test_2[i])
        x_xt[i, :] = np.add(x_xt[i, :], x_train_2)

    return x_xt


def nn_1_numpy(train: nd, test: nd) -> nd:
    x_xt = distance_matrix_reduced_numpy(train, test)
    return np.argmin(x_xt, axis=1)


def nn_1_numpy_split(train: nd, test: nd, max_size: int) -> nd:
    num_train = train.shape[0]
    num_test = test.shape[0]

    if max_size:
        test_batch_size = max_size // num_train

        if num_test % test_batch_size == 0:
            num_iters = num_test // test_batch_size
        else:
            num_iters = (num_test + test_batch_size - 1) // test_batch_size

        result = np.empty((num_test,), dtype=np.int64)

        for i in range(num_iters):
            start_index = i * test_batch_size
            end_index = np.minimum((i + 1) * test_batch_size, num_test)
            partial_result = nn_1_numpy(train, test[start_index:end_index, :])
            result[start_index:end_index] = partial_result

        return result

    else:
        return nn_1_numpy(train, test)


def compute_distance_matrix_loo(x: nd) -> nd:
    if x.dtype != np.float32:
        x = np.float32(x)

    x_xt = np.matmul(x, np.transpose(x))
    diag = np.diag(x_xt)
    d = np.copy(x_xt)

    for i in range(np.shape(d)[0]):
        d[i, :] = np.multiply(d[i, :], -2)
        d[i, :] = np.add(d[i, :], x_xt[i, i])
        d[i, :] = np.add(d[i, :], diag)
        d[i, i] = float("inf")

    return d


# Adapted from:
# https://en.wikipedia.org/wiki/Kruskal%27s_algorithm
def kruskal_mst(distance_matrix: np.ndarray) -> List[Tuple[int, int]]:
    shape = distance_matrix.shape
    assert len(shape) == 2 and shape[0] == shape[1], "Provided distance matrix should be a 2-d square matrix"

    ds = DisjointSet()
    sorted_edges_start, sorted_edges_end = np.unravel_index(np.argsort(distance_matrix, axis=None),
                                                            distance_matrix.shape)
    list_of_edges = []

    print("Sorting done")
    for edge_index in range(sorted_edges_start.size):
        if (edge_index + 1) % 1_000_000 == 0:
            print(f"Progress: {edge_index + 1}/{sorted_edges_start.size} "
                  f"({(edge_index + 1) * 100 / sorted_edges_start.size: .2f} %)", end="\r")
        edge_start = sorted_edges_start[edge_index]
        edge_end = sorted_edges_end[edge_index]

        if ds.find(edge_start) == ds.find(edge_end):
            continue
        else:
            ds.union(edge_start, edge_end)
            list_of_edges.append((edge_start, edge_end))

    return list_of_edges


def apply_label_noise(labels: nd, noise: float) -> nd:
    noisy_labels = np.copy(labels)

    if noise > 0.0:
        label_values = np.unique(noisy_labels)
        flip_indices = np.nonzero(np.random.binomial(n=1, p=noise, size=noisy_labels.size))[0]
        new_labels = np.random.choice(label_values, size=flip_indices.size)
        noisy_labels[flip_indices] = new_labels

    return noisy_labels


def compute_all_errs(data, split: str, noise: bool = False, ghp: bool = True) -> dict:
    if not noise:
        noise_values = [0.0]
    else:
        noise_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    result = {}

    # Load all data
    train_features = data["train_features"]
    train_labels = data["train_labels"].reshape(-1)
    test_features = data["test_features"]
    test_labels = data["test_labels"].reshape(-1)
    train_features_normalized = train_features / np.linalg.norm(train_features, axis=1).reshape(-1, 1)
    test_features_normalized = test_features / np.linalg.norm(test_features, axis=1).reshape(-1, 1)

    print("\tComputing 1-NN")
    indices_1nn = nn_1_numpy_split(train_features, test_features,
                                   max_size=train_labels.size * min(1_000, test_labels.size))

    print("\tComputing cosine 1-NN")
    indices_1nn_cosine = nn_1_numpy_split(train_features_normalized, test_features_normalized,
                                          max_size=train_labels.size * min(1_000, test_labels.size))

    if split == "train":
        features = train_features
        features_normalized = train_features_normalized
        num_samples = train_labels.size
        del test_features, test_features_normalized
    else:
        features = test_features
        features_normalized = test_features_normalized
        num_samples = test_labels.size
        del train_features, test_features_normalized

    print("\tComputing 1-NN LOO")
    d = compute_distance_matrix_loo(features)
    indices_1nn_loo = np.argmin(d, axis=1)

    print("\tComputing cosine 1-NN LOO")
    d_cosine = compute_distance_matrix_loo(features_normalized)
    indices_1nn_cosine_loo = np.argmin(d_cosine, axis=1)
    del d_cosine

    print("\tComputing GHP")
    mst_edges = None
    if ghp:
        mst_edges = kruskal_mst(d)
    del d

    # Prepare variables
    classes, _ = np.unique(train_labels, return_counts=True)
    num_test_samples = test_labels.size
    num_classes = len(classes)
    mapping = {c: i for (i, c) in enumerate(classes)}

    for noise_value in noise_values:
        if noise_value == 0.0:
            num_reps = 1
        else:
            num_reps = 30

        ghp_upper = []
        ghp_lower = []
        nn_loo = []
        nn_cosine_loo = []
        nn = []
        nn_cosine = []

        for repetition in range(num_reps):
            # Generate noisy labels
            noisy_train_labels = apply_label_noise(train_labels, noise_value)
            noisy_test_labels = apply_label_noise(test_labels, noise_value)

            if split == "train":
                noisy_labels = noisy_train_labels
            else:
                noisy_labels = noisy_test_labels

            if ghp:
                deltas = [[0.0] * (num_classes - i - 1) for i in range(num_classes - 1)]

                # Calculate number of dichotomous edges
                for i in range(noisy_labels.size - 1):
                    label_1 = mapping[noisy_labels[mst_edges[i][0]]]
                    label_2 = mapping[noisy_labels[mst_edges[i][1]]]
                    if label_1 == label_2:
                        continue
                    if label_1 > label_2:
                        label_1, label_2 = label_2, label_1
                    deltas[label_1][label_2 - label_1 - 1] += 1

                # Divide the number of dichotomous edges by 2 * num_train_samples to get estimator of deltas
                deltas = [[item / (2.0 * num_samples) for item in sublist] for sublist in deltas]

                # Sum up all the deltas
                delta_sum = sum([sum(sublist) for sublist in deltas])

                ghp_upper.append(2.0 * delta_sum)
                ghp_lower.append(((num_classes - 1.0) / float(num_classes)) * (
                        1.0 - math.sqrt(max(0.0, 1.0 - ((2.0 * num_classes) / (num_classes - 1.0) * delta_sum)))))

            # NN LOO
            predicted_labels_loo = noisy_labels[indices_1nn_loo]
            nn_loo.append(float(np.sum(noisy_labels != predicted_labels_loo)) / num_samples)

            # Cosine NN LOO
            predicted_labels_cosine_loo = noisy_labels[indices_1nn_cosine_loo]
            nn_cosine_loo.append(float(np.sum(noisy_labels != predicted_labels_cosine_loo)) / num_samples)

            # NN
            predicted_labels = noisy_train_labels[indices_1nn]
            nn.append(float(np.sum(noisy_test_labels != predicted_labels)) / num_test_samples)

            # Cosine NN
            predicted_labels_cosine = noisy_train_labels[indices_1nn_cosine]
            nn_cosine.append(float(np.sum(noisy_test_labels != predicted_labels_cosine)) / num_test_samples)

        result[noise_value] = {
            "GHP Upper": ghp_upper,
            "GHP Lower": ghp_lower,
            "1-NN": nn,
            "1-NN cosine": nn_cosine,
            "1-NN LOO": nn_loo,
            "1-NN cosine LOO": nn_cosine_loo
        }

    return result


if __name__ == "__main__":
    path = "results/"
    filenames = list(
        filter(lambda x: ".npz" in x,
               map(lambda x: os.path.join(path, x.name),
                   filter(lambda x: x.is_file(), os.scandir(path))))
    )

    filenames_str = "\n\t".join(filenames)
    print(f"Analyzing: \n\t{filenames_str}")

    for index, filename in enumerate(filenames):
        print(f"Evaluating embedding {index + 1}/{len(filenames)}")
        data_ = np.load(filename)
        print(f"\tLoading {filename} successful")
        dataset_and_embedding_name = filename.split(".")[0].split("/")[-1] + "-" + "test"
        result_path = os.path.join(path, f"{dataset_and_embedding_name}.txt")
        print(f"\tResults will be stored to: {result_path}")
        res_ = compute_all_errs(data_, split="test", noise=True, ghp=True)

        with open(result_path, "w") as f:
            f.write(json.dumps(res_, indent=4))

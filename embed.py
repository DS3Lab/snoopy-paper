import sys
from collections import OrderedDict

import torch as pt
from tensorflow_datasets import Split

from snoopy import set_cache_dir
from snoopy.embedding import *
from snoopy.pipeline import store_embeddings
from snoopy.reader import TFDSImageConfig, CSVFileConfig, TFDSTextConfig, NumpyArrayConfig

# Cache folder for datasets (TFDS) and embeddings (TensorFlow, PyTorch and HuggingFace)
cache_dir = "cache/"

# Folder where embeddings will be stored
# Files in the folder will look like "<dataset name>-<embedding name>.npz"
store_path = "results/"

# Path to the YELP dataset
path_yelp_train = "yelp/yelp_train.csv"
path_yelp_test = "yelp/yelp_test.csv"

# Path to the CIFAR-N dataset
path_cifar_n_features = "/mnt/ds3lab-scratch/rengglic/matrices_snoopy/{0}/{1}/features_raw.npy"
path_cifar_n_labels = "/mnt/ds3lab-scratch/rengglic/matrices_snoopy/{0}/{1}/labels_raw.npy"

if __name__ == "__main__":
    set_cache_dir(cache_dir)

    # Dataset to process
    dataset_name = sys.argv[1]

    # Index of embedding
    index = int(sys.argv[2])

    assert dataset_name in {"cifar10", "cifar100", "mnist", "yelp", "sst2", "imdb_reviews", "cifar10-aggre", "cifar10-worst", "cifar10-random1", "cifar10-random2", "cifar10-random3", "cifar100-noisy"}

    if dataset_name in {"cifar10-aggre", "cifar10-worst", "cifar10-random1", "cifar10-random2", "cifar10-random3", "cifar100-noisy"}:
        train_data = NumpyArrayConfig(path_features=path_cifar_n_features.format(dataset_name, 'train'),
                                      path_labels=path_cifar_n_labels.format(dataset_name, 'train'),
                                      height=32,
                                      width=32,
                                      num_channels=3)
        test_data = NumpyArrayConfig(path_features=path_cifar_n_features.format(dataset_name, 'test'),
                                     path_labels=path_cifar_n_labels.format(dataset_name, 'test'),
                                     height=32,
                                     width=32,
                                     num_channels=3)

    elif dataset_name in {"cifar10", "cifar100", "mnist"}:
        train_data = TFDSImageConfig(dataset_name=dataset_name, split=Split.TRAIN)
        test_data = TFDSImageConfig(dataset_name=dataset_name, split=Split.TEST)

    elif dataset_name == "imdb_reviews":
        train_data = TFDSTextConfig(dataset_name=dataset_name, split=Split.TRAIN)
        test_data = TFDSTextConfig(dataset_name=dataset_name, split=Split.TEST)

    elif dataset_name == "sst2":
        train_data = TFDSTextConfig(dataset_name="glue/sst2", split=Split.TRAIN, keys=("sentence", "label"))
        test_data = TFDSTextConfig(dataset_name="glue/sst2", split=Split.VALIDATION, keys=("sentence", "label"))

    # YELP dataset
    else:
        train_data = CSVFileConfig(
            path=path_yelp_train,
            header_present=False,
            text_column_number=1,
            label_column_number=2,
            num_columns=2,
            num_records=500_000,
            label_values=["1", "2", "3", "4", "5"],
            shuffle_buffer_size=500_000
        )
        test_data = CSVFileConfig(
            path=path_yelp_test,
            header_present=False,
            text_column_number=1,
            label_column_number=2,
            num_columns=2,
            num_records=50_000,
            label_values=["1", "2", "3", "4", "5"],
            shuffle_buffer_size=50_000
        )

    image_embeddings = OrderedDict({
        "alexnet": EmbeddingConfig(alexnet, batch_size=200, prefetch_size=10),
        "googlenet": EmbeddingConfig(googlenet, batch_size=200, prefetch_size=10),
        "vgg16": EmbeddingConfig(vgg16, batch_size=50, prefetch_size=10),
        "vgg19": EmbeddingConfig(vgg19, batch_size=50, prefetch_size=10),
        "inception": EmbeddingConfig(inception, batch_size=100, prefetch_size=10),
        "resnet_50_v2": EmbeddingConfig(resnet_50_v2, batch_size=100, prefetch_size=10),
        "resnet_101_v2": EmbeddingConfig(resnet_101_v2, batch_size=100, prefetch_size=10),
        "resnet_152_v2": EmbeddingConfig(resnet_152_v2, batch_size=100, prefetch_size=10),
        "efficientnet_b0": EmbeddingConfig(efficientnet_b0, batch_size=50, prefetch_size=10),
        "efficientnet_b1": EmbeddingConfig(efficientnet_b1, batch_size=50, prefetch_size=10),
        "efficientnet_b2": EmbeddingConfig(efficientnet_b2, batch_size=50, prefetch_size=10),
        "efficientnet_b3": EmbeddingConfig(efficientnet_b3, batch_size=50, prefetch_size=10),
        "efficientnet_b4": EmbeddingConfig(efficientnet_b4, batch_size=25, prefetch_size=10),
        "efficientnet_b5": EmbeddingConfig(efficientnet_b5, batch_size=10, prefetch_size=10),
        "efficientnet_b6": EmbeddingConfig(efficientnet_b6, batch_size=10, prefetch_size=10),
        "efficientnet_b7": EmbeddingConfig(efficientnet_b7, batch_size=5, prefetch_size=10),
    })

    text_embeddings = OrderedDict({
        "nnlm_50": EmbeddingConfig(nnlm_50, batch_size=1000, prefetch_size=10),
        "nnlm_50_normalization": EmbeddingConfig(nnlm_50_normalization, batch_size=1000, prefetch_size=10),
        "nnlm_128": EmbeddingConfig(nnlm_128, batch_size=1000, prefetch_size=10),
        "nnlm_128_normalization": EmbeddingConfig(nnlm_128_normalization, batch_size=1000, prefetch_size=10),
        "elmo": EmbeddingConfig(elmo, batch_size=4, prefetch_size=10),
        "use": EmbeddingConfig(use, batch_size=1000, prefetch_size=10),
        "use_large": EmbeddingConfig(use_large, batch_size=20, prefetch_size=10),
        "bert_cased_pool": EmbeddingConfig(bert_cased_pool, batch_size=50, prefetch_size=10),
        "bert_uncased_pool": EmbeddingConfig(bert_uncased_pool, batch_size=50, prefetch_size=10),
        "bert_cased": EmbeddingConfig(bert_cased, batch_size=50, prefetch_size=10),
        "bert_uncased": EmbeddingConfig(bert_uncased, batch_size=50, prefetch_size=10),
        "bert_cased_large_pool": EmbeddingConfig(bert_cased_large_pool, batch_size=25, prefetch_size=10),
        "bert_uncased_large_pool": EmbeddingConfig(bert_uncased_large_pool, batch_size=25, prefetch_size=10),
        "bert_cased_large": EmbeddingConfig(bert_cased_large, batch_size=25, prefetch_size=10),
        "bert_uncased_large": EmbeddingConfig(bert_uncased_large, batch_size=25, prefetch_size=10),
        "xlnet": EmbeddingConfig(xlnet, batch_size=10, prefetch_size=10),
        "xlnet_large": EmbeddingConfig(xlnet_large, batch_size=10, prefetch_size=10),
    })

    if dataset_name in {"cifar10", "cifar100", "cifar10-aggre", "cifar10-worst", "cifar10-random1", "cifar10-random2", "cifar10-random3", "cifar100-noisy"}:
        embeddings = image_embeddings
        embeddings["raw"] = EmbeddingConfig(ImageReshapeSpec(target_image_size=(32, 32), num_channels=3),
                                            batch_size=10, prefetch_size=10)
        embeddings["pca_32"] = EmbeddingConfig(PCASpec(output_dimension=32, target_image_size=(32, 32)),
                                               batch_size=50_000, prefetch_size=10)
        embeddings["pca_64"] = EmbeddingConfig(PCASpec(output_dimension=64, target_image_size=(32, 32)),
                                               batch_size=50_000, prefetch_size=10)
        embeddings["pca_128"] = EmbeddingConfig(PCASpec(output_dimension=128, target_image_size=(32, 32)),
                                                batch_size=50_000, prefetch_size=10)

    elif dataset_name == "mnist":
        embeddings = image_embeddings
        embeddings["raw"] = EmbeddingConfig(ImageReshapeSpec(target_image_size=(28, 28), num_channels=1),
                                            batch_size=10, prefetch_size=10)
        embeddings["pca_32"] = EmbeddingConfig(PCASpec(output_dimension=32, target_image_size=(28, 28)),
                                               batch_size=60_000, prefetch_size=10)
        embeddings["pca_64"] = EmbeddingConfig(PCASpec(output_dimension=64, target_image_size=(28, 28)),
                                               batch_size=60_000, prefetch_size=10)
        embeddings["pca_128"] = EmbeddingConfig(PCASpec(output_dimension=128, target_image_size=(28, 28)),
                                                batch_size=60_000, prefetch_size=10)

    else:
        embeddings = text_embeddings

    embedding_str = list(embeddings.keys())[index]
    device = pt.device("cpu")
    if not "pca" in embedding_str:
        device = pt.device("cuda:0")

    store_embeddings(train_data_config=train_data,
                     test_data_config=test_data,
                     embedding_configs=OrderedDict({embedding_str: embeddings[embedding_str]}),
                     device=device,
                     output_files_path=store_path,
                     filename_mapping={embedding_str: dataset_name + "-" + embedding_str})

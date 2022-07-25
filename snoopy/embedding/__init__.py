from .base import EmbeddingConfig, EmbeddingDataset, EmbeddingDatasetsTuple, EmbeddingIterator, EmbeddingModelSpec
from .dummy import DummySpec, ImageReshapeSpec
from .hugging_face import HuggingFaceSpec
from .sklearn_transform import PCASpec
from .tf_hub import TFHubImageSpec, TFHubTextSpec
from .torch_hub import TorchHubImageSpec

# Image embeddings
inception = TFHubImageSpec(url="https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/4",
                           output_dimension=2048, required_image_size=(299, 299))

vgg19 = TorchHubImageSpec(name="vgg19", output_dimension=4096, layer_extractor=lambda x: x.classifier[5],
                          required_image_size=(224, 224))

vgg16 = TorchHubImageSpec(name="vgg16", output_dimension=4096, layer_extractor=lambda x: x.classifier[5],
                          required_image_size=(224, 224))

alexnet = TorchHubImageSpec(name="alexnet", output_dimension=4096, layer_extractor=lambda x: x.classifier[5],
                            required_image_size=(224, 224))

googlenet = TorchHubImageSpec(name="googlenet", output_dimension=1024, layer_extractor=lambda x: x.dropout,
                              required_image_size=(224, 224))

resnet_50_v2 = TFHubImageSpec(url="https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/4",
                              output_dimension=2048, required_image_size=(224, 224))

resnet_101_v2 = TFHubImageSpec(url="https://tfhub.dev/google/imagenet/resnet_v2_101/feature_vector/4",
                               output_dimension=2048, required_image_size=(224, 224))

resnet_152_v2 = TFHubImageSpec(url="https://tfhub.dev/google/imagenet/resnet_v2_152/feature_vector/4",
                               output_dimension=2048, required_image_size=(224, 224))

efficientnet_b0 = TFHubImageSpec(url="https://tfhub.dev/tensorflow/efficientnet/b0/feature-vector/1",
                                 output_dimension=1280, required_image_size=(224, 224))

efficientnet_b1 = TFHubImageSpec(url="https://tfhub.dev/tensorflow/efficientnet/b1/feature-vector/1",
                                 output_dimension=1280, required_image_size=(240, 240))

efficientnet_b2 = TFHubImageSpec(url="https://tfhub.dev/tensorflow/efficientnet/b2/feature-vector/1",
                                 output_dimension=1408, required_image_size=(260, 260))

efficientnet_b3 = TFHubImageSpec(url="https://tfhub.dev/tensorflow/efficientnet/b3/feature-vector/1",
                                 output_dimension=1536, required_image_size=(300, 300))

efficientnet_b4 = TFHubImageSpec(url="https://tfhub.dev/tensorflow/efficientnet/b4/feature-vector/1",
                                 output_dimension=1792, required_image_size=(380, 380))

efficientnet_b5 = TFHubImageSpec(url="https://tfhub.dev/tensorflow/efficientnet/b5/feature-vector/1",
                                 output_dimension=2048, required_image_size=(456, 456))

efficientnet_b6 = TFHubImageSpec(url="https://tfhub.dev/tensorflow/efficientnet/b6/feature-vector/1",
                                 output_dimension=2304, required_image_size=(528, 528))

efficientnet_b7 = TFHubImageSpec(url="https://tfhub.dev/tensorflow/efficientnet/b7/feature-vector/1",
                                 output_dimension=2560, required_image_size=(600, 600))

# Image embeddings - other
mobilenet = TorchHubImageSpec(name="mobilenet_v2", output_dimension=1280,
                              layer_extractor=lambda x: x.classifier[0], required_image_size=(224, 224))

resnet_152_torch = TorchHubImageSpec(name="resnet152", output_dimension=2048, layer_extractor=lambda x: x.avgpool,
                                     required_image_size=(224, 224))

# Text embeddings
elmo = TFHubTextSpec(url="https://tfhub.dev/google/elmo/3", output_dimension=1024)

nnlm_50 = TFHubTextSpec(url="https://tfhub.dev/google/nnlm-en-dim50/2", output_dimension=50)

nnlm_50_normalization = TFHubTextSpec(url="https://tfhub.dev/google/nnlm-en-dim50-with-normalization/2",
                                      output_dimension=50)

nnlm_128 = TFHubTextSpec(url="https://tfhub.dev/google/nnlm-en-dim128/2", output_dimension=128)

nnlm_128_normalization = TFHubTextSpec(url="https://tfhub.dev/google/nnlm-en-dim128-with-normalization/2",
                                       output_dimension=128)

use = TFHubTextSpec(url="https://tfhub.dev/google/universal-sentence-encoder/4", output_dimension=512)

use_large = TFHubTextSpec(url="https://tfhub.dev/google/universal-sentence-encoder-large/5", output_dimension=512)

bert_cased = HuggingFaceSpec(name="bert-base-cased", output_dimension=768, max_length=512, fast_tokenizer=True)

bert_uncased = HuggingFaceSpec(name="bert-base-uncased", output_dimension=768, max_length=512, fast_tokenizer=True)

bert_cased_large = HuggingFaceSpec(name="bert-large-cased", output_dimension=1024, max_length=512, fast_tokenizer=True)

bert_uncased_large = HuggingFaceSpec(name="bert-large-uncased", output_dimension=1024, max_length=512,
                                     fast_tokenizer=True)

bert_cased_pool = HuggingFaceSpec(name="bert-base-cased", output_dimension=768, max_length=512, fast_tokenizer=True,
                                  pool=True)

bert_uncased_pool = HuggingFaceSpec(name="bert-base-uncased", output_dimension=768, max_length=512, fast_tokenizer=True,
                                    pool=True)

bert_cased_large_pool = HuggingFaceSpec(name="bert-large-cased", output_dimension=1024, max_length=512,
                                        fast_tokenizer=True, pool=True)

bert_uncased_large_pool = HuggingFaceSpec(name="bert-large-uncased", output_dimension=1024, max_length=512,
                                          fast_tokenizer=True, pool=True)

xlnet = HuggingFaceSpec(name="xlnet-base-cased", output_dimension=768, max_length=512)

xlnet_large = HuggingFaceSpec(name="xlnet-large-cased", output_dimension=1024, max_length=512)

# Text embeddings - other
openai_gpt = HuggingFaceSpec(name="openai-gpt", output_dimension=768, max_length=512,
                             tokenizer_params={"pad_token": "<pad>"})

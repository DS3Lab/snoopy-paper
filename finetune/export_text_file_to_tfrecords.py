from absl import app
from absl import flags
from absl import logging
import numpy as np

import tensorflow as tf
import tensorflow_datasets as tfds

from transformers import TFBertForSequenceClassification
from transformers import BertTokenizer

import os.path as path
import json

import time
import itertools

FLAGS = flags.FLAGS

# can be up to 512 for BERT
flags.DEFINE_integer('max_length', 512, 'Max sequence length')

flags.DEFINE_string("file_path", "data_yelp", "Path to the features and labels file")
flags.DEFINE_string("file_json", "yelp_academic_dataset_review.json", "JSON file containing features and labels")
flags.DEFINE_string("file_export_train", "yelp_train.tfrecord", "TFRecord file containing train features and labels")
flags.DEFINE_string("file_export_test", "yelp_test.tfrecord.", "TFRecord file containing test features and labels")
flags.DEFINE_string("file_dict_features", 'text', "Property containing the features")
flags.DEFINE_string("file_dict_labels", 'stars', "Property containing the labels")
flags.DEFINE_integer("file_startindex_train", 0, "Start index of the entries in the file")
flags.DEFINE_integer("file_endindex_train", 500000, "End index of the entries in the file")
flags.DEFINE_integer("file_startindex_test", 500000, "Start index of the entries in the file")
flags.DEFINE_integer("file_endindex_test", 550000, "End index of the entries in the file")

def test_argument_and_file(folder_path, arg_name):
    if arg_name not in FLAGS.__flags.keys() or not FLAGS.__flags[arg_name].value:
        raise app.UsageError("--{} is a required argument when runing the tool.".format(arg_name))

    arg_val = FLAGS.__flags[arg_name].value

    if not path.exists(path.join(folder_path, arg_val)):
        raise app.UsageError("File '{}' given by '--{}' does not exists in the specified folder '{}'.".format(arg_val, arg_name, folder_path))

def load_and_log_json():
    logging.log(logging.DEBUG, "Start loading file '{}'".format(FLAGS.file_json))
    start = time.time()
    with open(path.join(FLAGS.file_path, FLAGS.file_json)) as f:
        content = f.readlines()
    content_train = content[FLAGS.file_startindex_train:FLAGS.file_endindex_train]
    content_test = content[FLAGS.file_startindex_test:FLAGS.file_endindex_test]
    features_train = []
    labels_train = []
    features_test = []
    labels_test = []
    samples_train = len(content_train)
    samples_test = len(content_test)
    for line in content_train:
        review = json.loads(line)
        features_train.append(review[FLAGS.file_dict_features])
        labels_train.append(int(review[FLAGS.file_dict_labels]))
    for line in content_test:
        review = json.loads(line)
        features_test.append(review[FLAGS.file_dict_features])
        labels_test.append(int(review[FLAGS.file_dict_labels]))
    end = time.time()
    logging.log(logging.DEBUG, "'{}' loaded in {} seconds".format(FLAGS.file_json, end - start))

    logging.log(logging.INFO, "'{}' loaded as lists with '{}' training and '{}' test samples".format(FLAGS.file_json, samples_train, samples_test))
    return features_train, labels_train, samples_train, features_test, labels_test, samples_test

def main(argv):

    print("Version: ", tf.__version__)
    print("Eager mode: ", tf.executing_eagerly())
    print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")

    test_argument_and_file(FLAGS.file_path, "file_json")

    def feature_fn(text: tf.Tensor) -> tf.Tensor:
        return text

    model_name = 'bert-base-uncased'
    #model_name = 'bert-large-uncased'

    tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=True)

    def convert_example_to_feature(review):
        # combine step for tokenization, WordPiece vector mapping, adding special tokens as well as truncating reviews longer than the max length

        return tokenizer.encode_plus(review, 
                add_special_tokens = True, # add [CLS], [SEP]
                max_length = FLAGS.max_length, # max length of the text that can go to BERT
                pad_to_max_length = True, # add [PAD] tokens
                return_attention_mask = True, # add attention mask to not focus on pad tokens
                truncation=True,
                )

    # map to the expected input to TFBertForSequenceClassification, see here
    def map_example_to_dict(input_ids, attention_masks, token_type_ids, label):
        return {
                "input_ids": input_ids,
                "token_type_ids": token_type_ids,
                "attention_mask": attention_masks,
                }, label

    def encode_examples(features, labels, save_file, limit=-1):
        # prepare list, so that we can build up final TensorFlow dataset from slices.
        input_ids_list = []
        token_type_ids_list = []
        attention_mask_list = []
        label_list = []
        if (limit > 0):
            features = features[:limit]
            labels = labels[:limit]

        for i, (review, label) in enumerate(zip(features, labels)):
            if i % 10000 == 0:
                print("  Processing sample", i)
            bert_input = convert_example_to_feature(review)
      
            input_ids_list.append(bert_input['input_ids'])
            token_type_ids_list.append(bert_input['token_type_ids'])
            attention_mask_list.append(bert_input['attention_mask'])
            label_list.append([label])

        ds = tf.data.Dataset.from_tensor_slices((input_ids_list, attention_mask_list, token_type_ids_list, label_list))

        def write_map_fn(x1, x2, x3, x4):
            return tf.io.serialize_tensor(tf.concat([x1, x2, x3, x4], -1))
        ds = ds.map(write_map_fn)

        writer = tf.data.experimental.TFRecordWriter(save_file)
        writer.write(ds)

        return

        ## Read
        #def read_map_fn(x):
        #    xp = tf.io.parse_tensor(x, tf.int32)
        #    # Optionally set shape
        #    xp.set_shape([1537])  # Do `xp.set_shape([None, 1537])` if using batches
        #    # Use `x[:, :512], ...` if using batches
        #    return xp[:512], xp[512:1024], xp[1024:1536], xp[-1:]
        #ds = tf.data.TFRecordDataset('mydata.tfrecord').map(read_map_fn)
        #print(ds)

        #return ds.map(map_example_to_dict)

    features_train, labels_train, samples_train, features_test, labels_test, samples_test = load_and_log_json()

    accs = []
    times = []
    print("Encoding for saving training samples...")
    encode_examples(features_train, labels_train, path.join(FLAGS.file_path, FLAGS.file_export_train))
    print("Encoding for saving test samples...")
    encode_examples(features_test, labels_test, path.join(FLAGS.file_path, FLAGS.file_export_test))

if __name__ == '__main__':
  app.run(main)

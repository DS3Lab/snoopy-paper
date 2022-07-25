from absl import app
from absl import logging
from absl import flags
import numpy as np

import tensorflow as tf
import tensorflow_datasets as tfds

import autokeras as ak

from random import random, randint

import os.path as path
import json
import time
import itertools

FLAGS = flags.FLAGS

flags.DEFINE_string("dataset", "yelp", "Dataset name for output")
flags.DEFINE_string("file_path", "/nfs/iiscratch-zhang.inf.ethz.ch/export/systems/export/rengglic/data_yelp", "Path to the features and labels file")
flags.DEFINE_string("file_json", "yelp_academic_dataset_review.json", "JSON file containing features and labels")
flags.DEFINE_string("file_export_train", "yelp_train.tfrecord", "TFRecord file containing train features and labels")
flags.DEFINE_string("file_export_test", "yelp_test.tfrecord.", "TFRecord file containing test features and labels")
flags.DEFINE_string("file_dict_features", 'text', "Property containing the features")
flags.DEFINE_string("file_dict_labels", 'stars', "Property containing the labels")
flags.DEFINE_integer("file_startindex_train", 0, "Start index of the entries in the file")
flags.DEFINE_integer("file_endindex_train", 500000, "End index of the entries in the file")
flags.DEFINE_integer("file_startindex_test", 500000, "Start index of the entries in the file")
flags.DEFINE_integer("file_endindex_test", 550000, "End index of the entries in the file")
flags.DEFINE_integer('num_classes', 5, 'Number of classes')
flags.DEFINE_float('noise', 0.0, 'Label noise fraction')
flags.DEFINE_string('project_name_suffix', None, "Suffix to automatically generated project name")
flags.DEFINE_string('output', None, 'output file name.')

flags.DEFINE_integer('epochs', 100, 'Number of epochs')
flags.DEFINE_integer('max_trials', 2, 'Number of max trials')

def test_argument_and_file(folder_path, arg_name):
    if arg_name not in FLAGS.__flags.keys() or not FLAGS.__flags[arg_name].value:
        raise app.UsageError("--{} is a required argument when runing the tool.".format(arg_name))

    arg_val = FLAGS.__flags[arg_name].value

    if not path.exists(path.join(folder_path, arg_val)):
        raise app.UsageError("File '{}' given by '--{}' does not exists in the specified folder '{}'.".format(arg_val, arg_name, folder_path))

def load_and_log_json(noise_level, num_labels):
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

    rnd = 0

    for line in content_train:
        review = json.loads(line)
        features_train.append(review[FLAGS.file_dict_features])
        label = int(review[FLAGS.file_dict_labels]) - 1
        if random() < noise_level:
            rnd = rnd + 1
            label = randint(0,num_labels-1)
        #if tf.random.uniform([1])[0] < noise_level:
        #    rnd = rnd + 1
        #    # First two parameters are irrelevant in this case!
        #    label = tf.random.uniform_candidate_sampler(
        #        true_classes=[[0]], num_true=1, num_sampled=1, unique=True, range_max=num_labels
        #    ).sampled_candidates[0].numpy()
        labels_train.append(label)
    for line in content_test:
        review = json.loads(line)
        features_test.append(review[FLAGS.file_dict_features])
        label = int(review[FLAGS.file_dict_labels]) - 1
        if random() < noise_level:
            rnd = rnd + 1
            label = randint(0,num_labels-1)
        #if tf.random.uniform([1])[0] < noise_level:
        #    rnd = rnd + 1
        #    # First two parameters are irrelevant in this case!
        #    label = tf.random.uniform_candidate_sampler(
        #        true_classes=[[0]], num_true=1, num_sampled=1, unique=True, range_max=num_labels
        #    ).sampled_candidates[0].numpy()
        labels_test.append(label)
    end = time.time()
    logging.log(logging.DEBUG, "'{}' loaded in {} seconds".format(FLAGS.file_json, end - start))

    logging.log(logging.INFO, "'{}' loaded as lists with '{}' training and '{}' test samples (and {} random labels inthere)".format(FLAGS.file_json, samples_train, samples_test, rnd))
    return np.array(features_train), np.array(labels_train), np.array(samples_train), np.array(features_test), np.array(labels_test), samples_test

def main(argv):

    print("Version: ", tf.__version__)
    print("Eager mode: ", tf.executing_eagerly())
    print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")

    test_argument_and_file(FLAGS.file_path, "file_json")

    features_train, labels_train, samples_train, features_test, labels_test, samples_test = load_and_log_json(FLAGS.noise, FLAGS.num_classes)

    ds_train = tf.data.Dataset.from_tensor_slices(((features_train,), (labels_train,)))
    ds_test = tf.data.Dataset.from_tensor_slices(((features_test,), (labels_test,)))

    if FLAGS.project_name_suffix:
        project_name = "{}_noise_{}_{}".format(FLAGS.dataset, FLAGS.noise, FLAGS.project_name_suffix)
    else:
        project_name = "{}_noise_{}".format(FLAGS.dataset, FLAGS.noise)

    clf = ak.TextClassifier(project_name=project_name, directory="/cluster/scratch/rengglic/output_autokeras/models", max_trials=FLAGS.max_trials)
    start = time.time()
    # Feed the tensorflow Dataset to the classifier.
    clf.fit(ds_train, epochs=FLAGS.epochs)
    end = time.time()
    # Evaluate the best model with testing data.
    acc = clf.evaluate(ds_test)
    print("Final accuracy: ", acc)
    print('Noise: {}'.format(FLAGS.noise))
    print("Duration overall:", end - start)

    if FLAGS.output:
        with open(FLAGS.output, "w+") as f:
            f.write('noise: {}\n'.format(FLAGS.noise))
            f.write("accuracy: {}\n".format(acc[1]))
            f.write("time: {}\n".format(end - start))

if __name__ == '__main__':
  app.run(main)

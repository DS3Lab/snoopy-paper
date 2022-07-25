from absl import app
from absl import flags
import numpy as np

import tensorflow as tf
import tensorflow_datasets as tfds

import autokeras as ak

import time
import itertools

FLAGS = flags.FLAGS

flags.DEFINE_string('dataset', "mnist", 'TFDS dataset.')
flags.DEFINE_float('noise', 0.0, 'Label noise fraction')
flags.DEFINE_string('project_name_suffix', None, "Suffix to automatically generated project name")
flags.DEFINE_string('output', None, 'output file name.')

flags.DEFINE_integer('epochs', 100, 'Number of epochs')
flags.DEFINE_integer('max_trials', 2, 'Number of max trials')
flags.DEFINE_float('added_time', 0.0, 'Time to add to this experiment')

def main(argv):

    print("Version: ", tf.__version__)
    print("Eager mode: ", tf.executing_eagerly())
    print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")

    (ds_train, ds_test), ds_info = tfds.load(
        FLAGS.dataset,
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )
    num_classes = ds_info.features["label"].num_classes

    def label_noise(label: tf.Tensor, num_labels: int, noise_level: float) -> tf.Tensor:
        if tf.random.uniform([1])[0] < noise_level:
            # First two parameters are irrelevant in this case!
            return tf.random.uniform_candidate_sampler(
                true_classes=[[0]], num_true=1, num_sampled=1, unique=True, range_max=num_labels
            ).sampled_candidates[0]
        else:
            return label

    def preparation_fn_with_label_noise(feature: tf.Tensor, label: tf.Tensor):
        return feature, label_noise(label, num_classes, FLAGS.noise)

    print("Pre-processing train...")

    ds_train = ds_train.map(preparation_fn_with_label_noise)

    print("Pre-processing test...")

    ds_test = ds_test.map(preparation_fn_with_label_noise)

    if FLAGS.project_name_suffix:
        project_name = "{}_noise_{}_{}".format(FLAGS.dataset, FLAGS.noise, FLAGS.project_name_suffix)
    else:
        project_name = "{}_noise_{}".format(FLAGS.dataset, FLAGS.noise)

    clf = ak.ImageClassifier(project_name=project_name, directory="/cluster/scratch/rengglic/output_autokeras/models", max_trials=FLAGS.max_trials)

    start = time.time()
    # Feed the tensorflow Dataset to the classifier.
    clf.fit(ds_train, epochs=FLAGS.epochs)
    end = time.time()
    # Evaluate the best model with testing data.
    acc = clf.evaluate(ds_test)
    print("Final accuracy: ", acc)
    print('Noise: {}'.format(FLAGS.noise))
    print("Duration overall:", (end - start)+FLAGS.added_time)

    if FLAGS.output:
        with open(FLAGS.output, "w+") as f:
            f.write('noise: {}\n'.format(FLAGS.noise))
            f.write("accuracy: {}\n".format(acc[1]))
            f.write("time: {}\n".format((end - start)+FLAGS.added_time))

if __name__ == '__main__':
    app.run(main)

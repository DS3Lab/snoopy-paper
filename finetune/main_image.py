from absl import app
from absl import flags
import numpy as np

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds

import time
import itertools
#import threading

FLAGS = flags.FLAGS

flags.DEFINE_string('dataset', "cifar100", 'TFDS dataset.')
flags.DEFINE_string('output', None, 'output file name.')
flags.DEFINE_float('noise', 0.0, 'Label noise fraction')
flags.DEFINE_float('lr', 0.01, 'SGD learning rate')
flags.DEFINE_float('reg', 0.0, 'L2 reguralizer also known as weight decay')
flags.DEFINE_integer('img_size', 224, 'Image size to resize to before input')
flags.DEFINE_integer('epochs', 102, 'SGD epochs')
flags.DEFINE_integer('batch_size', 16, 'SGD mini-batch size')
flags.DEFINE_float('momentum', 0.9, 'SGD momentum')
flags.DEFINE_boolean('nesterov', True, 'Use nesterov momentum')
flags.DEFINE_boolean('fine_tune', True, 'Fine tune or freeze weights')
#flags.DEFINE_integer('epochs', 25, 'SGD epochs')
#flags.DEFINE_integer('batch_size', 8, 'SGD mini-batch size')
#flags.DEFINE_float('momentum', 0.0, 'SGD momentum')
#flags.DEFINE_boolean('nesterov', False, 'Use nesterov momentum')
#flags.DEFINE_boolean('fine_tune', False, 'Fine tune or freeze weights')


def main(argv):

    print("Version: ", tf.__version__)
    print("Eager mode: ", tf.executing_eagerly())
    print("Hub version: ", hub.__version__)
    print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")

    (ds_train, ds_test), ds_info = tfds.load(
        FLAGS.dataset,
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )
    num_classes = ds_info.features["label"].num_classes
    IMAGE_SIZE = FLAGS.img_size
    BATCH_SIZE = FLAGS.batch_size

    def feature_fn(image: tf.Tensor) -> tf.Tensor:
        return tf.image.resize(tf.cast(image, tf.float32) / 255., [IMAGE_SIZE, IMAGE_SIZE])

    def label_noise(label: tf.Tensor, num_labels: int, noise_level: float) -> tf.Tensor:
        if tf.random.uniform([1])[0] < noise_level:
            # First two parameters are irrelevant in this case!
            return tf.random.uniform_candidate_sampler(
                true_classes=[[0]], num_true=1, num_sampled=1, unique=True, range_max=num_labels
            ).sampled_candidates[0]
        else:
            return label

    def preparation_fn_with_label_noise(feature: tf.Tensor, label: tf.Tensor):
        return feature_fn(feature), label_noise(label, num_classes, FLAGS.noise)

    print("Pre-processing train...")

    ds_train = ds_train.map(preparation_fn_with_label_noise)
    #ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_train = ds_train.shuffle(10000)
    ds_train = ds_train.batch(BATCH_SIZE)

    print("Pre-processing test...")

    ds_test = ds_test.map(preparation_fn_with_label_noise)
    ds_test = ds_test.batch(BATCH_SIZE)

    print("Laoding hub module...")

    loaded_obj = hub.load("https://tfhub.dev/tensorflow/efficientnet/b4/feature-vector/1")

    m = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
        hub.KerasLayer(loaded_obj, trainable=FLAGS.fine_tune),
        tf.keras.layers.Dense(num_classes, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(FLAGS.reg))
    ])
    m.build([None, IMAGE_SIZE, IMAGE_SIZE, 3])  # Batch input shape.

    #m.summary()

    m.compile(
        optimizer=tf.keras.optimizers.SGD(lr=FLAGS.lr, momentum=FLAGS.momentum, nesterov=FLAGS.nesterov), 
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy'],
    )
     
    epochs = FLAGS.epochs
    start = time.time()
    m.fit(
        ds_train,
        epochs=epochs,
        validation_data=ds_test,
        verbose=0,
    )
    end = time.time()
    print("Epochs (", epochs, ") Duration overall:", end - start)
    print("Epochs (", epochs, ") Duration average:", (end - start) / epochs)

    acc = m.evaluate(ds_test, verbose=0)
    print("Final accuracy: ", acc)
    print('noise: {}'.format(FLAGS.noise))
    print('learning rate: {}'.format(FLAGS.lr))
    print('l2 regularizer: {}'.format(FLAGS.reg))
    print("epochs: {}".format(epochs))
    print("time: {}".format(end - start))
    
    if FLAGS.output:
        with open(FLAGS.output, "w+") as f:
            f.write('noise: {}\n'.format(FLAGS.noise))
            f.write("accuracy: {}\n".format(acc[1]))
            f.write('learning rate: {}\n'.format(FLAGS.lr))
            f.write('l2 regularizer: {}\n'.format(FLAGS.reg))
            f.write("epochs: {}\n".format(epochs))
            f.write("time: {}\n".format(end - start))

if __name__ == '__main__':
    app.run(main)

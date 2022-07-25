from absl import app
from absl import flags
from absl import logging
import numpy as np

import tensorflow as tf
import tensorflow_datasets as tfds

from transformers import TFBertForSequenceClassification
from transformers import BertTokenizer
#from transformers.configuration_bert import BertConfig

import os.path as path
import json

import time
import itertools

FLAGS = flags.FLAGS

flags.DEFINE_string('output', None, 'output file name.')
flags.DEFINE_float('noise', 0.0, 'Label noise fraction')
# can be up to 512 for BERT
flags.DEFINE_integer('max_length', 512, 'Max sequence length')
# recommended learning rate for Adam 5e-5, 3e-5, 2e-5
flags.DEFINE_float('lr', 2e-5, 'SGD learning rate')
flags.DEFINE_integer('epochs', 1, 'SGD epochs')
flags.DEFINE_integer('batch_size', 6, 'SGD mini-batch size')
flags.DEFINE_boolean('validate', False, 'Validate after each epoch')

flags.DEFINE_integer('num_classes', 5, 'Number of classes')
flags.DEFINE_string("file_path", "/nfs/iiscratch-zhang.inf.ethz.ch/export/systems/export/rengglic/data_yelp", "Path to the features and labels file")
flags.DEFINE_string("file_export_train", "yelp_train.tfrecord", "TFRecord file containing train features and labels")
flags.DEFINE_string("file_export_test", "yelp_test.tfrecord.", "TFRecord file containing test features and labels")

def main(argv):

    print("Version: ", tf.__version__)
    print("Eager mode: ", tf.executing_eagerly())
    print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")

    def feature_fn(text: tf.Tensor) -> tf.Tensor:
        return text

    def label_noise(label: tf.Tensor, num_labels: int, noise_level: float) -> tf.Tensor:
        if tf.random.uniform([1])[0] < noise_level:
            # First two parameters are irrelevant in this case!
            return tf.cast(tf.random.uniform_candidate_sampler(
                true_classes=[[0]], num_true=1, num_sampled=1, unique=True, range_max=num_labels
            ).sampled_candidates[0], dtype=tf.int32)
        else:
            return label

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
                }, label_noise(label-1, FLAGS.num_classes, FLAGS.noise)

    def get_ds(save_file):

        # Read
        def read_map_fn(x):
            xp = tf.io.parse_tensor(x, tf.int32)
            # Optionally set shape
            xp.set_shape([1537])  # Do `xp.set_shape([None, 1537])` if using batches
            # Use `x[:, :512], ...` if using batches
            return map_example_to_dict(xp[:512], xp[512:1024], xp[1024:1536], xp[-1])
        ds = tf.data.TFRecordDataset(save_file).map(read_map_fn)
        return ds

    print("Get datastes")
    ds_train = get_ds(path.join(FLAGS.file_path, FLAGS.file_export_train)).shuffle(10000).batch(FLAGS.batch_size)
    ds_test = get_ds(path.join(FLAGS.file_path, FLAGS.file_export_test)).batch(FLAGS.batch_size)

    #for example in ds_train:
    #    print(example[0]["input_ids"].numpy()[0,:])
    #    print(example[0]["token_type_ids"].numpy()[0,:])
    #    print(example[0]["attention_mask"].numpy()[0,:])
    #    print(example[1].numpy())
    #    return

    #config = BertConfig("bert_config.json")
    #config["num_labels"] = FLAGS.num_classes
    #print(config)

    model = TFBertForSequenceClassification.from_pretrained(model_name, num_labels=FLAGS.num_classes)
    
    #optimizer = tf.keras.optimizers.Adam(learning_rate=FLAGS.lr, epsilon=1e-08)
    optimizer = tf.keras.optimizers.Adam(learning_rate=FLAGS.lr)

    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
    model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

    print(f'Training model with {model_name}')
    start = time.time()
    if FLAGS.validate:
        bert_history = model.fit(ds_train, epochs=FLAGS.epochs, validation_data=ds_test, verbose=2)
    else:
        bert_history = model.fit(ds_train, epochs=FLAGS.epochs, verbose=2)
    end = time.time()
    print("Long (", FLAGS.epochs, ") Duration overall:", end - start)
    print("Long (", FLAGS.epochs, ") Duration average:", (end - start) / FLAGS.epochs)

    loss, acc = model.evaluate(ds_test, verbose=0)
    print("Final accuracy: ", acc)
    print('noise: {}'.format(FLAGS.noise))
    print('learning rate: {}'.format(FLAGS.lr))
    print("epochs: {}".format(FLAGS.epochs))
    print("time: {}".format(end - start))

    if FLAGS.output:
        with open(FLAGS.output, "w+") as f:
            f.write('noise: {}\n'.format(FLAGS.noise))
            f.write('learning rate: {}\n'.format(FLAGS.lr))
            f.write("accuracy: {}\n".format(acc))
            f.write("epochs: {}\n".format(FLAGS.epochs))
            f.write("time: {}\n".format(end - start))

if __name__ == '__main__':
  app.run(main)

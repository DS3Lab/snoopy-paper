from absl import app
from absl import flags
import numpy as np

import tensorflow as tf
import tensorflow_datasets as tfds

from transformers import TFBertForSequenceClassification
from transformers import BertTokenizer

import time
import itertools

FLAGS = flags.FLAGS

flags.DEFINE_string('dataset', 'imdb_reviews', 'output file name.')
flags.DEFINE_string('test_split', 'test', 'output file name.')
flags.DEFINE_string('feature', 'text', 'output file name.')
flags.DEFINE_string('label', 'label', 'output file name.')
flags.DEFINE_string('output', None, 'output file name.')
flags.DEFINE_float('noise', 0.0, 'Label noise fraction')
# can be up to 512 for BERT
flags.DEFINE_integer('max_length', 512, 'Max sequence length')
# recommended learning rate for Adam 5e-5, 3e-5, 2e-5
flags.DEFINE_float('lr', 2e-5, 'SGD learning rate')
flags.DEFINE_integer('epochs', 1, 'SGD epochs')
flags.DEFINE_integer('batch_size', 6, 'SGD mini-batch size')
flags.DEFINE_boolean('validate', False, 'Validate after each epoch')

def main(argv):

    print("Version: ", tf.__version__)
    print("Eager mode: ", tf.executing_eagerly())
    print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")

    def feature_fn(text: tf.Tensor) -> tf.Tensor:
        return text

    def label_noise(label: tf.Tensor, num_labels: int, noise_level: float) -> tf.Tensor:
        if tf.random.uniform([1])[0] < noise_level:
            # First two parameters are irrelevant in this case!
            return tf.random.uniform_candidate_sampler(
                true_classes=[[0]], num_true=1, num_sampled=1, unique=True, range_max=num_labels
            ).sampled_candidates[0]
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
                }, label

    def encode_examples(ds, limit=-1):
        # prepare list, so that we can build up final TensorFlow dataset from slices.
        input_ids_list = []
        token_type_ids_list = []
        attention_mask_list = []
        label_list = []
        if (limit > 0):
            ds = ds.take(limit)
      
        for x in tfds.as_numpy(ds):
            review = x[FLAGS.feature]
            label = x[FLAGS.label]
            bert_input = convert_example_to_feature(review.decode())
      
            input_ids_list.append(bert_input['input_ids'])
            token_type_ids_list.append(bert_input['token_type_ids'])
            attention_mask_list.append(bert_input['attention_mask'])
            label_list.append([label_noise(label, num_classes, FLAGS.noise)])
        return tf.data.Dataset.from_tensor_slices((input_ids_list, attention_mask_list, token_type_ids_list, label_list)).map(map_example_to_dict)

    (ds_train, ds_test), ds_info = tfds.load(
        "glue/{}".format(FLAGS.dataset) if FLAGS.dataset in ["sst2"] else FLAGS.dataset,
        split=['train', FLAGS.test_split],
        shuffle_files=True,
        with_info=True,
    )
    num_classes = ds_info.features["label"].num_classes
    ds_train = encode_examples(ds_train).shuffle(10000).batch(FLAGS.batch_size)
    ds_test = encode_examples(ds_test).batch(FLAGS.batch_size)

    model = TFBertForSequenceClassification.from_pretrained(model_name)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=FLAGS.lr, epsilon=1e-08)

    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
    model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

    print(f'Training model with {model_name}')
    start = time.time()
    if FLAGS.validate:
        bert_history = model.fit(ds_train, epochs=FLAGS.epochs, validation_data=ds_test, verbose=0)
    else:
        bert_history = model.fit(ds_train, epochs=FLAGS.epochs, verbose=0)
    end = time.time()
    print("Epochs (", FLAGS.epochs, ") Duration overall:", end - start)
    print("Epochs (", FLAGS.epochs, ") Duration average:", (end - start) / FLAGS.epochs)

    loss, acc = model.evaluate(ds_test, verbose=0)
    print("Final accuracy: ", acc)
    print('noise: {}'.format(FLAGS.noise))
    print('learning rate: {}'.format(FLAGS.lr))
    print("epochs: {}".format(FLAGS.epochs))
    print("time: {}".format(end - start))

    if FLAGS.output:
        with open(FLAGS.output, "w+") as f:
            f.write('noise: {}\n'.format(FLAGS.noise))
            f.write("accuracy: {}\n".format(acc))
            f.write("epochs: {}\n".format(FLAGS.epochs))
            f.write("time: {}\n".format(end - start))

if __name__ == '__main__':
  app.run(main)

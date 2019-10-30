!pip install wget
import wget
import zipfile
import tarfile
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import sys

import tarfile
from six.moves import cPickle as pickle
from six.moves import xrange
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


dataset_config = [
    "CIFAR10",
    "MNIST",
    "IMAGENET",
    "TINYIMAGENET"
]


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def cifar10_loader():
    CIFAR_FILENAME = 'cifar-10-python.tar.gz'
    CIFAR_DOWNLOAD_URL = 'https://www.cs.toronto.edu/~kriz/' + CIFAR_FILENAME
    CIFAR_LOCAL_FOLDER = 'cifar-10-batches-py'


    def download_and_extract(data_dir):
      # download CIFAR-10 if not already downloaded.
      tf.contrib.learn.datasets.base.maybe_download(CIFAR_FILENAME, data_dir,
                                                    CIFAR_DOWNLOAD_URL)
      tarfile.open(os.path.join(data_dir, CIFAR_FILENAME),
                  'r:gz').extractall(data_dir)

    def _get_file_names():
      """Returns the file names expected to exist in the input_dir."""
      file_names = {}
      file_names['train'] = ['data_batch_%d' % i for i in xrange(1, 5)]
      file_names['validation'] = ['data_batch_5']
      file_names['eval'] = ['test_batch']
      return file_names


    def read_pickle_from_file(filename):
      with tf.gfile.Open(filename, 'rb') as f:
        if sys.version_info >= (3, 0):
          data_dict = pickle.load(f, encoding='bytes')
        else:
          data_dict = pickle.load(f)
      return data_dict


    def convert_to_tfrecord(input_files, output_file):
      """Converts a file to TFRecords."""
      print('Generating %s' % output_file)
      with tf.python_io.TFRecordWriter(output_file) as record_writer:
        for input_file in input_files:
          data_dict = read_pickle_from_file(input_file)
          data = data_dict[b'data']
          labels = data_dict[b'labels']
          num_entries_in_batch = len(labels)
          for i in range(num_entries_in_batch):
            example = tf.train.Example(features=tf.train.Features(
                feature={
                    'image': _bytes_feature(data[i].tobytes()),
                    'label': _int64_feature(labels[i])
                }))
            record_writer.write(example.SerializeToString())

    data_dir = "./"
    print('Download from {} and extract.'.format(CIFAR_DOWNLOAD_URL))
    download_and_extract(data_dir)
    file_names = _get_file_names()
    input_dir = os.path.join(data_dir, CIFAR_LOCAL_FOLDER)
    for mode, files in file_names.items():
      input_files = [os.path.join(input_dir, f) for f in files]
      output_file = os.path.join(data_dir, mode + '.tfrecords')
      try:
        os.remove(output_file)
      except OSError:
        pass
      # Convert to tf.train.Example and write the to TFRecords.
      convert_to_tfrecord(input_files, output_file)
    print('Done! CIFAR10 is now present at your working directory')


def mnist_loader():
    def _data_path(data_directory:str, name:str) -> str:

      if not os.path.isdir(data_directory):
          os.makedirs(data_directory)

      return os.path.join(data_directory, f'{name}.tfrecords')

    def _int64_feature(value:int) -> tf.train.Features.FeatureEntry:
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _bytes_feature(value:str) -> tf.train.Features.FeatureEntry:
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def convert_to(data_set, name:str, data_directory:str, num_shards:int=1):
        print(f'Processing {name} data')

        images = data_set.images
        labels = data_set.labels
        
        num_examples, rows, cols, depth = data_set.images.shape

        def _process_examples(start_idx:int, end_index:int, filename:str):
            with tf.python_io.TFRecordWriter(filename) as writer:
                for index in range(start_idx, end_index):
                    sys.stdout.write(f"\rProcessing sample {index+1} of {num_examples}")
                    sys.stdout.flush()

                    image_raw = images[index].tostring()
                    example = tf.train.Example(features=tf.train.Features(feature={
                        'height': _int64_feature(rows),
                        'width': _int64_feature(cols),
                        'depth': _int64_feature(depth),
                        'label': _int64_feature(int(labels[index])),
                        'image_raw': _bytes_feature(image_raw)
                    }))
                    writer.write(example.SerializeToString())
        
        if num_shards == 1:
            _process_examples(0, data_set.num_examples, _data_path(data_directory, name))
        else:
            total_examples = data_set.num_examples
            samples_per_shard = total_examples // num_shards

            for shard in range(num_shards):
                start_index = shard * samples_per_shard
                end_index = start_index + samples_per_shard
                _process_examples(start_index, end_index, _data_path(data_directory, f'{name}-{shard+1}'))

        print()

    def convert_to_tf_record(data_directory:str):

        mnist = input_data.read_data_sets(
            "/tmp/tensorflow/mnist/input_data", 
            reshape=False
        )
        
        convert_to(mnist.validation, 'validation', data_directory)
        convert_to(mnist.train, 'train', data_directory, num_shards=10)
        convert_to(mnist.test, 'test', data_directory)

    convert_to_tf_record(os.path.expanduser("./"))


def download_dataset(dataset_name):
    if dataset_name not in dataset_config:
        print(list(dataset_config.keys()))
        msg = "Please enter dataset name from the above dataset list"
        return msg
    else:
        print("Downloading your dataset...")
    
    if(dataset_name == "CIFAR10"):
      return cifar10_loader()

    if(dataset_name == "MNIST"):
      return mnist_loader()

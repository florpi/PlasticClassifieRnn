"""
inputs
References:
    https://github.com/tensorflow/models/blob/1f484095c0981e2a62403b16256cb877749dfe94/research/object_detection/inputs.py
"""
import os
import json
import functools
import tensorflow as tf

# From our repo
import utils
import decode_tfrecords 
import preprocess

NUM_BANDS = decode_tfrecords.NUM_BANDS
PLASTICC_CATEGORIES = [6, 15 ,16 ,42 ,52 ,53 ,62 ,64 ,65 ,67 ,88 ,90 ,92 ,95 ,99]
MAPPER_CLASSIFIER_PLASTICC = {cat:idx for idx, cat in enumerate(PLASTICC_CATEGORIES)}
NUM_CATEGORIES = len(PLASTICC_CATEGORIES)

SERVING_FED_EXAMPLE_KEY = 'serialized_example'

def _get_features_dict(input_dict, train_stats=None):
    """
    Preprocess input features.
    Args:
        input_dict
    Returns:
        features: dict containing object_id, static, and dynamic features.
        Where dynamic features contains temporal features for band_0, band_1 ... band_5.
        features['object_id'] : tensor of shape []
        features['dynamic']   : features['dynamic']['flux_0']
                                features['dynamic']['flux_err_0']
                               ...
    """
    features = {'object_id':input_dict['object_id']}
    encoder_feats = {}
    for band in range(NUM_BANDS):
        # Add time differences
        features['band_%i/time_diff'%band] = tf.concat([[0], 
            input_dict['band_%i/mjd'%band][1:] - input_dict['band_%i/mjd'%band][:-1]], axis=0)
        # Keep track of number of samples
        features['band_%i/num_samples'%band] = input_dict['band_%i/num_samples'%band]
        features['band_%i/mjd'%band] = input_dict['band_%i/mjd'%band]
        features['band_%i/augmented_flux'%band] = input_dict['band_%i/flux'%band]
        features['band_%i/flux_err'%band] = input_dict['band_%i/flux_err'%band]
        if 'band_%i/original_flux'%band in input_dict.keys():
            features['band_%i/original_flux'%band] = input_dict['band_%i/original_flux'%band]

    # Flux normalization
    flux_list = [features['band_%i/augmented_flux'%band] for band in range(NUM_BANDS)]
    preprocessed_fluxes = preprocess._standard_normalize(flux_list)
    for band, flux in enumerate(preprocessed_fluxes):
        features['band_%i/preprocessed_flux'%band] = flux

    # time diff normalization
    time_list = [features['band_%i/time_diff'%band] for band in range(NUM_BANDS)]
    preprocessed_time = preprocess._max_min_normalize(time_list)
    for band, time_diff in enumerate(preprocessed_time):
        features['band_%i/preprocessed_time_diff'%band] = time_diff

    # temporal features
    stacked_times = [tf.stack([features['band_%i/time_diff'%band],
                                input_dict['band_%i/flux'%band],
                                input_dict['band_%i/flux_err'%band],
                                tf.to_float(input_dict['band_%i/detected'%band])], axis = -1) for band in range(NUM_BANDS)]

    # dft periodogram features
    stacked_dfts = [tf.stack([input_dict['band_%i/dft/freqs'%band],
                              input_dict['band_%i/dft/mag'%band],
                              input_dict['band_%i/dft/phase'%band],
                              input_dict['band_%i/dft/periodogram'%band],
                              input_dict['band_%i/dft/proba'%band]], axis=-1) for band in range(NUM_BANDS)]

    #preprocessed_dfts = stacked_dfts
    # Normalize Fourier features
    preprocessed_times = preprocess._standard_normalize(stacked_times)

    for band, band_time in enumerate(preprocessed_times):
        # Stack dft (mag ,phase..) features
        features['band_%i/times'%band] = band_time

    preprocessed_dfts = preprocess._standard_normalize(stacked_dfts)

    for band, band_dft in enumerate(preprocessed_dfts):
        # Stack dft (mag ,phase..) features
        features['band_%i/dft'%band] = band_dft
        # dft signals have their own num of samples
        features['band_%i/dft/num_samples'%band] = tf.shape(input_dict['band_%i/dft/freqs'%band])[0]


    return features

def _get_labels_dict(input_dict):
    """
    Optionally provides weightmaps
    """
    return {'target':input_dict['target']}

def _augment(tensor_dict):
    """
    Data augmentation scheme for train_input_fn
    """
    # Embed each band signal in additive gaussian noise, where std=flux_err
    for band in range(NUM_BANDS):
        tensor_dict['band_%i/original_flux'%band] = tensor_dict['band_%i/flux'%band]
        tensor_dict['band_%i/flux'%band] = preprocess._add_gaussian_noise(tensor_dict['band_%i/flux'%band],
                                                                     std=tensor_dict['band_%i/flux_err'%band])
    return tensor_dict

def create_train_input_fn(dataset_dir, validation_fold, batch_size=1, train_stats=None):
    """Creates a train 'input' function for 'Estimator'.
    Args:

    Returns:
        'input_fn' for 'Estimator' in TRAIN mode.
    """

    def _train_input_fn(params=None):
        """Returns 'features' and 'labels' tensor dictionaries for training.
        Args:
          params: Parameter dictionary passed from the estimator.
        Returns:
          A tf.data.Dataset that holds (features, labels) tuple.
        """
        def transform_fn(tensor_dict):
            tensor_dict = _augment(tensor_dict)
            return (_get_features_dict(tensor_dict, train_stats), _get_labels_dict(tensor_dict))

        dataset = build_dataset(validation_fold, dataset_dir, batch_size, 
                                is_training=True, transform_input_data_fn=transform_fn)
        return dataset
    return _train_input_fn

def create_eval_input_fn(dataset_dir, batch_size=1, train_stats=None):
    """Creates a eval 'input' function for 'Estimator'.
    Args:

    Returns:
        'input_fn' for 'Estimator' in TRAIN mode.
    """

    def _eval_input_fn(params=None):
        """Returns 'features' and 'labels' tensor dictionaries for training.
        Args:
          params: Parameter dictionary passed from the estimator.
        Returns:
          A tf.data.Dataset that holds (features, labels) tuple.
        """
        def transform_fn(tensor_dict):
            return (_get_features_dict(tensor_dict, train_stats), _get_labels_dict(tensor_dict))

        dataset = build_dataset(validation_fold, dataset_dir, batch_size,
                                is_training=False, transform_input_data_fn=transform_fn)
        return dataset

    return _eval_input_fn

def create_predict_input_fn(train_stats=None):
    """Creates a predict input function for Estimator.
    Args:
      predict_input_config: An input_reader_pb2.InputReader.
    Returns:
      'input_fn' for Estimator in PREDICT mode.
    """

    def _predict_input_fn(params=None):
        """Decodes serialized tf.Examples and returns 'ServingInputReceiver'.
        Args:
          params: Parameter dictionary passed from the estimator.
        Returns:
          'ServingInputReceiver'.
        """
        del params
        example = tf.placeholder(dtype=tf.string, shape=[], name='tf_example')

        decoder = decode_tfrecords.TfExampleDecoder(0, '')
        input_dict = decoder.decode(example)
        feats_dict = _get_features_dict(input_dict, train_stats)
        # Batch items
        feats_dict = {feat_name:tf.expand_dims(feat, 0) for feat_name, feat in feats_dict.items()}
        return tf.estimator.export.ServingInputReceiver(features=feats_dict,
                                                        receiver_tensors={SERVING_FED_EXAMPLE_KEY: example})
    return _predict_input_fn

def read_dataset(file_read_func, filenames, num_readers=64, shuffle=True, num_epochs=None):
    """Reads a dataset, and handles repetition and shuffling.

    Args:
      file_read_func: Function to use in tf.contrib.data.parallel_interleave, to
        read every individual file into a tf.data.Dataset.
      input_files: A list of file paths to read.
      config: A input_reader_builder.InputReader object.

    Returns:
      A tf.data.Dataset of (undecoded) tf-records based on config.
    """
    buffer_size = 6000
    # Shard, shuffle, and read files.
    if num_readers > len(filenames):
        num_readers = len(filenames)
        tf.logging.warning('num_readers has been reduced to %d to match input file shards.' % num_readers)
    filename_dataset = tf.data.Dataset.from_tensor_slices(filenames)
    
    if shuffle:
        filename_dataset = filename_dataset.shuffle(buffer_size)
    elif num_readers > 1:
        tf.logging.warning('shuffle is false, but the input data stream is still slightly shuffled since num_readers > 1.')
    filename_dataset = filename_dataset.repeat(num_epochs)
    records_dataset = filename_dataset.apply(tf.contrib.data.parallel_interleave(file_read_func,
                                                                                 cycle_length=num_readers,
                                                                                 sloppy=shuffle))
    if shuffle:
        records_dataset = records_dataset.shuffle(buffer_size)
    return records_dataset

def build_dataset(validation_fold, dataset_dir, batch_size=1, is_training=True,
                  transform_input_data_fn=None, num_prefetch_batches=64):
    """Builds a tf.data.Dataset.

    Builds a tf.data.Dataset by applying the 'transform_input_data_fn' on all
    records. Applies a padded batch to the resulting dataset.

    features: dict containing object_id, static_feats, band_0, band_1 ... band_5

    Args:
      validation_fold: fold in which to perform validation.
      input_reader_config: A input_reader_pb2.InputReader object.
      batch_size: Batch size. If batch size is None, no batching is performed.
      transform_input_data_fn: Function to apply transformation to all records,
        or None if no extra decoding is required.

    Returns:
      A tf.data.Dataset based on the input_reader_config.

    Raises:
      ValueError: On invalid input reader proto.
      ValueError: If no input paths are specified.
    """

    decoder = decode_tfrecords.TfExampleDecoder(validation_fold, dataset_dir)

    def process_fn(value):
        """Sets up tf graph that decodes, transforms and pads input data."""
        processed_tensors = decoder.decode(value)
        if transform_input_data_fn is not None:
            processed_tensors = transform_input_data_fn(processed_tensors)
        return processed_tensors

    dataset = read_dataset(functools.partial(tf.data.TFRecordDataset,
                                             buffer_size=64 * 1000 * 1000),
                           decoder.filenames,
                           num_epochs=None if is_training else 1)
    # One parallel call per batch, batches are decoded in parallel
    num_parallel_calls = batch_size

    dataset = dataset.map(process_fn,
                          num_parallel_calls=num_parallel_calls)

    # Make batch padding everything with 0
    features_padded_shapes = {}
    for band in range(NUM_BANDS):

        features_padded_shapes.update( {'band_%i/preprocessed_flux'%band: [None],
                                          'band_%i/mjd'%band: [None],
                                          'band_%i/time_diff'%band: [None],
                                          'band_%i/preprocessed_time_diff'%band: [None],
                                          'band_%i/augmented_flux'%band: [None],
                                          'band_%i/flux_err'%band: [None],
                                          'band_%i/dft'%band: [None, 6],
                                          'band_%i/times'%band: [None, 4],
                                          'band_%i/dft/num_samples'%band:[],
                                          'band_%i/num_samples'%band:[]})
        if is_training:
            # Keep track of original flux (without augmentation)
            features_padded_shapes.update({'band_%i/original_flux'%band: [None]})

    features_padded_shapes['object_id'] = []

    dataset = dataset.padded_batch(batch_size,
                                   # Padded shapes, still without batch!
                                   padded_shapes=((features_padded_shapes),
                                                  ({'target':[]})),
                                   drop_remainder=True)

    dataset = dataset.prefetch(num_prefetch_batches)
    return dataset

def read_metadata(dataset_dir):
    """
    Read dataset metadata (train_moments, class_weights...) if available.
    """
    try:
        with open(os.path.join(dataset_dir, 'metadata.json')) as metadata:
            tf.logging.info('Reading dataset metadata.')
            return json.load(metadata)
    except FileNotFoundError:
        return {}

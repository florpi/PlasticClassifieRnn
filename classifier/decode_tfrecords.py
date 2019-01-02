# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Provides data from Plasticc shynthetic dataset.
"""
import collections
import glob
import os.path
import tensorflow as tf

slim = tf.contrib.slim

dataset = slim.dataset

tfexample_decoder = slim.tfexample_decoder

# Default file pattern of TFRecord of TensorFlow Example.

# Number of bands in Plasticc dataset
NUM_BANDS = 6

class TfExampleDecoder():
    """
    """
    def __init__(self, validation_fold, dataset_dir, is_training=True):
        """Init decoder of segmentation tfrecords.

        Args:
          validation_fold: A train/val Split name.
          dataset_dir: The directory of the dataset sources.

        Returns:
          An instance of slim Dataset.

        """
        if is_training:
            filenames = glob.glob(os.path.join(dataset_dir,'*.tfrecord'))
            # Select all files that are not validation for training
            filenames = [f for f in filenames if int(f.split('fold_')[-1].split('_')[0]) \
                    != validation_fold]

        else:
            filenames = glob.glob(os.path.join(dataset_dir, '*.tfrecord'))
            filenames = [f for f in filenames if int(f.split('fold_')[-1].split('_')[0]) \
                    == validation_fold]


            filenames = [os.path.join(dataset_dir, filenames % validation_fold)]
        selected_folds = list(set([str(int(f.split('fold_')[-1].split('_')[0])) for f in filenames]))
        tf.logging.info('Building input pipeline for %s from folds: %s.'%('TRAIN' if is_training else 'EVAL',
                                                                          ', '.join(selected_folds)))
        ## Specify how the TF-Examples are decoded.
        # Timeless features (Fixed len)
        keys_to_features = {'object/id': tf.FixedLenFeature((), tf.float32, default_value=0),
                            'object/target': tf.FixedLenFeature((), tf.int64, default_value=0),
                            'ddf': tf.FixedLenFeature((), tf.int64, default_value=0),
                            'hostgal_specz': tf.FixedLenFeature((), tf.float32, default_value=0),
                            'hostgal_photoz': tf.FixedLenFeature((), tf.float32, default_value=0),
                            'hostgal_photoz_err': tf.FixedLenFeature((), tf.float32, default_value=0),
                            'distmod': tf.FixedLenFeature((), tf.float32, default_value=0),
                            'mwebv': tf.FixedLenFeature((), tf.float32, default_value=0)}


        for band in range(NUM_BANDS):
            # Time dependent features by band (Var len)
            keys_to_features.update({'band_%i/num_samples'%band: tf.FixedLenFeature((), tf.int64, default_value=0),
                                     'band_%i/detected'%band: tf.VarLenFeature(dtype=tf.int64),
                                     'band_%i/flux'%band: tf.VarLenFeature(dtype=tf.float32),
                                     'band_%i/flux_err'%band: tf.VarLenFeature(dtype=tf.float32),
                                     'band_%i/mjd'%band: tf.VarLenFeature(dtype=tf.float32),
                                     'band_%i/dft/freqs'%band: tf.VarLenFeature(dtype=tf.float32),
                                     'band_%i/dft/mag'%band: tf.VarLenFeature(dtype=tf.float32),
                                     'band_%i/dft/phase'%band: tf.VarLenFeature(dtype=tf.float32),
                                     'band_%i/dft/periodogram'%band: tf.VarLenFeature(dtype=tf.float32),
                                     'band_%i/dft/proba'%band: tf.VarLenFeature(dtype=tf.float32)})

        items_to_handlers = {'object_id': tfexample_decoder.Tensor('object/id'),
                             'target': tfexample_decoder.Tensor('object/target'),
                             'ddf': tfexample_decoder.Tensor('ddf'),
                             'hostgal_specz': tfexample_decoder.Tensor('hostgal_specz'),
                             'hostgal_photoz': tfexample_decoder.Tensor('hostgal_photoz'),
                             'hostgal_photoz_err': tfexample_decoder.Tensor('hostgal_photoz_err'),
                             'distmod': tfexample_decoder.Tensor('distmod'),
                             'mwebv': tfexample_decoder.Tensor('mwebv')}

        for band in range(NUM_BANDS):
            items_to_handlers.update({'band_%i/num_samples'%band: tfexample_decoder.Tensor('band_%i/num_samples'%band),
                                      'band_%i/detected'%band:  tfexample_decoder.Tensor('band_%i/detected'%band),
                                      'band_%i/flux'%band: tfexample_decoder.Tensor('band_%i/flux'%band),
                                      'band_%i/flux_err'%band: tfexample_decoder.Tensor('band_%i/flux_err'%band),
                                      'band_%i/mjd'%band: tfexample_decoder.Tensor('band_%i/mjd'%band),
                                      'band_%i/dft/freqs'%band: tfexample_decoder.Tensor('band_%i/dft/freqs'%band),
                                      'band_%i/dft/mag'%band: tfexample_decoder.Tensor('band_%i/dft/mag'%band),
                                      'band_%i/dft/phase'%band: tfexample_decoder.Tensor('band_%i/dft/phase'%band),
                                      'band_%i/dft/periodogram'%band: tfexample_decoder.Tensor('band_%i/dft/periodogram'%band),
                                      'band_%i/dft/proba'%band: tfexample_decoder.Tensor('band_%i/dft/proba'%band)})



        self.num_bands = NUM_BANDS
        self.decoder = tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)
        self.filenames = filenames

    def decode(self, tf_example_string_tensor):
        """
        """
        serialized_example = tf.reshape(tf_example_string_tensor, shape=[])
        keys = self.decoder.list_items()
        tensors = self.decoder.decode(serialized_example, items=keys)
        tensor_dict = dict(zip(keys, tensors))
        return tensor_dict

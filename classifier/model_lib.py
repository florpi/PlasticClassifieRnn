import os
import numpy as np
import tensorflow as tf
import tfplot
from tensorflow.python import debug as tf_debug

# From our repo
import utils
import  inputs
import rnn
import vis_utils

NUM_CATEGORIES = inputs.NUM_CATEGORIES
PLASTICC_CATEGORIES = inputs.PLASTICC_CATEGORIES
NUM_BANDS = inputs.NUM_BANDS
# Class weights from https://www.kaggle.com/c/PLAsTiCC-2018/discussion/67194
KAGGLE_WEIGHTS =  [1., 2., 1., 1., 1., 1., 1., 2., 1., 1., 1., 1., 1., 1.] # without 99 class

slim = tf.contrib.slim

def model_fn(features, labels, mode, params ):
    """
    Model function for tf.estimator.
    Args:
        model: Select model to use (FCN or RNN)
        features: Dictionary of feature tensors, returned from 'input_fn'.
        labels: Dictionary of groundtruth tensors if mode is TRAIN or EVAL, otherwise None.
        mode: Mode key from tf.estimator.ModeKeys.
        params={# The model must choose between 3 classes.
                'num_classes': 3}
    """

    # Get or create global step
    global_step = tf.train.get_or_create_global_step()

    if params['model'] == 'fourier':
        dft_features = [features['band_%i/dft'%band] for band in range(NUM_BANDS)]
        num_samples = [features['band_%i/dft/num_samples'%band] for band in range(NUM_BANDS)]

        embeddings = rnn.rnn_logits(dft_features, num_samples, mode==tf.estimator.ModeKeys.TRAIN)

        predictions = rnn.dense_logits(embeddings, NUM_CATEGORIES)

    elif params['model'] == 'time':
        time_features = [features['band_%i/times'%band] for band in range(NUM_BANDS)]
        num_samples = [features['band_%i/num_samples'%band] for band in range(NUM_BANDS)]

        embeddings = rnn.rnn_logits(time_features, num_samples, mode==tf.estimator.ModeKeys.TRAIN)
        predictions = rnn.dense_logits(embeddings, NUM_CATEGORIES)


    elif params['model'] == 'joint':
        dft_features = [features['band_%i/dft'%band] for band in range(NUM_BANDS)]
        num_samples = [features['band_%i/dft/num_samples'%band] for band in range(NUM_BANDS)]

        with tf.variable_scope("FourierEmbeddings"):
            dft_embeddings = rnn.rnn_logits(dft_features, num_samples,
                    mode == tf.estimator.ModeKeys.TRAIN)

        time_features = [features['band_%i/times'%band] for band in range(NUM_BANDS)]
        num_samples = [features['band_%i/num_samples'%band] for band in range(NUM_BANDS)]

        with tf.variable_scope("TemporalEmbeddings"):
            time_embeddings = rnn.rnn_logits(time_features, num_samples,
                    mode==tf.estimator.ModeKeys.TRAIN)

        embeddings = tf.concat([dft_embeddings,
                                time_embeddings,
                                tf.expand_dims(features['flux_range'], -1)],
                               axis=-1)
        fcn = rnn.fcn_logits(embeddings, mode==tf.estimator.ModeKeys.TRAIN)
        predictions = rnn.dense_logits(fcn, NUM_CATEGORIES)

    else:
        raise ValueError("Not implemented, choose ['fourier', 'time', 'joint']'. Selected %s"%params['model'])

    total_loss = None
    if mode in (tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL):
        # Compute weights
        weights = np.asarray(KAGGLE_WEIGHTS)
        if params['class_weights'] is not None:
            tf.logging.info('Weighting loss by class frequency.')
            weights *= np.asarray(params['class_weights'], dtype=np.float32)

        # Create loss
        rnn.create_loss(predictions, None, labels, weights, NUM_CATEGORIES, None, num_samples)
        # Get total loss
        total_loss = tf.losses.get_total_loss(add_regularization_losses=True, name='total_loss')

    train_op = None
    if mode == tf.estimator.ModeKeys.TRAIN:
        tf.logging.info('Creating train ops...')

        # Inputs summaries
        feats_values = list(features.values())[0:]
        feats_keys = list(features.keys())
        tfplot.summary.plot_many(plot_func=vis_utils.plot_light_curves, in_tensors=feats_values,
                                 name='inputs', max_outputs=2, feats_keys=feats_keys)


        # Add summaries for losses
        for loss in tf.get_collection(tf.GraphKeys.LOSSES):
            tf.summary.scalar('losses/%s' % loss.op.name, loss)

        train_op = tf.contrib.layers.optimize_loss(loss=total_loss,
                                                   global_step=global_step,
                                                   learning_rate=params.get('learning_rate', 1e-3),
                                                   clip_gradients=params.get('clip_gradients_value', 50.),
                                                   optimizer='Adam',
                                                   name='')  # Preventing scope prefix on all variables.

    eval_metric_ops = None
    eval_summary_hooks = None
    if mode == tf.estimator.ModeKeys.EVAL:
        tf.logging.info('Creating evaluation ops...')

        eval_metric_ops = {}
        ## Eval metricts
        # miou
        eval_metric_ops['accuracy'] = tf.metrics.accuracy(predictions['classes'], labels['target'])
        # loss
        for loss in tf.get_collection(tf.GraphKeys.LOSSES):
            eval_metric_ops['losses/%s'%loss.op.name] = tf.metrics.mean(loss)

        # Confusion matrix
        eval_metric_ops['mean_iou'] = tf.metrics.mean_iou(labels['target'], predictions['classes'], NUM_CATEGORIES)

        # Cast keys to string
        eval_metric_ops = {str(k): v for k, v in eval_metric_ops.items()}

        # Create a SummarySaverHook. It will catch eval summaries and save them propertly.
        eval_images_dir = os.path.join(params['model_dir'], 'eval_images')
        tf.logging.info('Saving evaluation images into %s'%eval_images_dir)
        confusionMatrixSaveHook = vis_utils.ConfusionMatrixSaverHook(labels=[str(i) for i in PLASTICC_CATEGORIES],
                                                                     confusion_matrix_tensor_name = 'mean_iou/total_confusion_matrix',
                                                                     summary_writer = tf.summary.FileWriterCache.get(eval_images_dir))
        eval_summary_hooks = [confusionMatrixSaveHook]

    export_outputs = None
    if mode == tf.estimator.ModeKeys.PREDICT:
        tf.logging.info('Creating prediction ops...')
        export_outputs = {tf.saved_model.signature_constants.PREDICT_METHOD_NAME:
                          tf.estimator.export.PredictOutput(predictions)}
    # Logout number of trainable params
    num_parameters = np.sum([np.prod(v.shape) for v in tf.trainable_variables()])
    tf.logging.info('Graph created with %i trainable params.'%num_parameters)

    return tf.estimator.EstimatorSpec(mode=mode,
                                      predictions=predictions,
                                      loss=total_loss,
                                      train_op=train_op,
                                      eval_metric_ops=eval_metric_ops,
                                      evaluation_hooks=eval_summary_hooks,
                                      export_outputs=export_outputs)


def create_estimator_and_inputs(model, run_config, dataset_dir, learning_rate, validation_fold, batch_size=1, num_epochs=None):
    """
    """
    # Read dataset metadata
    metadata = inputs.read_metadata(dataset_dir, validation_fold)
    train_stats = metadata.get('train_stats', None)

    # Create the input functions for TRAIN/EVAL/PREDICT.
    train_input_fn = inputs.create_train_input_fn(dataset_dir, validation_fold, batch_size=batch_size, train_stats=train_stats)
    eval_input_fn = inputs.create_eval_input_fn(dataset_dir, validation_fold, batch_size=batch_size, train_stats=train_stats)
    predict_input_fn = inputs.create_predict_input_fn(train_stats=train_stats)

    train_steps = None
    if num_epochs is not None and 'train_objects' in metadata.keys():
        train_steps = len(metadata['train_objects'])//batch_size*num_epochs
    # Create estimator
    estimator = tf.estimator.Estimator(model_fn=model_fn,
                                       params={'model':model,
                                               'model_dir':run_config.model_dir,
                                               'learning_rate':learning_rate,
                                               'class_weights':metadata.get('train_class_weights_sorted_list', None)},
                                       config=run_config)

    return dict(estimator=estimator,
                train_input_fn=train_input_fn,
                eval_input_fn=eval_input_fn,
                predict_input_fn=predict_input_fn,
                train_steps=train_steps)

def create_train_and_eval_specs(train_input_fn,
                                eval_input_fn,
                                predict_input_fn,
                                train_steps,
                                eval_secs=1,
                                final_exporter_name='Servo'):
    """Creates a 'TrainSpec' and 'EvalSpec's.
    Args:
      train_input_fn: Function that produces features and labels on train data.
      eval_input_fn: Functions that produce features and labels on eval data.
      predict_input_fn: Function that produces features for inference.
      train_steps: Number of training steps.
      final_exporter_name: String name given to 'FinalExporter'.
    Returns:
      Tuple of 'TrainSpec' and list of 'EvalSpecs'.
    """
    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=train_steps)
                                        #hooks = [tf_debug.LocalCLIDebugHook()])

    eval_spec_name = '1'
    exporter_name = '{}_{}'.format(final_exporter_name, eval_spec_name)
    exporter = tf.estimator.FinalExporter(name=exporter_name, serving_input_receiver_fn=predict_input_fn)
    #exporter = tf.estimator.BestExporter(name=exporter_name, serving_input_receiver_fn=predict_input_fn)
    eval_spec = tf.estimator.EvalSpec(name=eval_spec_name,
                                      input_fn=eval_input_fn,
                                      steps=None,
                                      throttle_secs=eval_secs,
                                      exporters=exporter)
    return train_spec, eval_spec

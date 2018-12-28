from collections import OrderedDict

import numpy as np
import tensorflow as tf
# From our repo
import  utils

slim = tf.contrib.slim

def create_loss(predictions, decoder_reconstruction, labels, class_weights, n_class, features, num_samples):
    """
    """
    with tf.name_scope('Loss'):
        # Gather class weights for particular dataset examples
        classification_weight = 1.0; reconstruction_weight = 2.0;
        class_weights *= classification_weight
        weights = tf.gather(class_weights, labels['target'])
        # Create classification and reconstruction losses
        create_classification_loss(predictions['scores'], labels['target'], weights, n_class)
        #create_reconstruction_loss(decoder_reconstruction, features, reconstruction_weight*weights, num_samples)

def create_classification_loss(predictions, labels, class_weights, n_class):
    """
    """
    with tf.name_scope('Classification'):
        
        one_hot_labels = slim.one_hot_encoding(labels, n_class, on_value=1.0, off_value=0.0)
        tf.losses.log_loss(one_hot_labels,
                           predictions,
                           weights=tf.expand_dims(class_weights, 1),
                           scope='LogLoss')

def create_reconstruction_loss(predictions, labels, class_weights, num_samples):
    """
    """
    with tf.name_scope('Reconstruction'):
        predictions = utils.pad_and_stack(predictions)
        labels = utils.pad_and_stack(labels)
        # Gather class weights for particular dataset examples
        seq_mask = utils.pad_and_stack([tf.expand_dims(tf.sequence_mask(n, dtype=tf.float32), -1) for n in num_samples]) # [bs, seq_len, n_feats, bands]
        weights = tf.reshape(class_weights, [-1, 1, 1, 1])*seq_mask
        tf.losses.huber_loss(labels, predictions, weights=weights, delta=1.0, scope='huber')

def _rnn_net(inputs, hidden_sizes, last_relevant, initial_states, keep_prob=None,
             reuse=None, direction='bidirectional'):
    """
    Creates RNN layer.
    Args:
        hidden_sizes : array with size of the hidden size of each layer

    Returns:
        net: last time output
    """
    def batch_gather(tensor, ind):
        """
        Get specified elements along the first axis of tensor.
        Args:
            tensor: tensor to be sliced.
            ind: indices to take (one for each element along axis 0 of data).
        returns: sliced tensor.
        """
        with tf.name_scope('GatherLastRelevant'):
            batch_range = tf.range(tensor.shape.as_list()[0])
            indices = tf.stack([batch_range, tf.cast(ind, tf.int32)], axis=1)
            return tf.gather_nd(tensor, indices)

    def rnn_cell(hidden_size, keep_prob):
        """
        Creates RNN cell and optionally use dropout between cells
        """
        #cell = tf.nn.rnn_cell.BasicRNNCell(hidden_size, reuse=tf.get_variable_scope().reuse)
        cell = tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell(hidden_size, reuse=tf.get_variable_scope().reuse)
        #cell = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(hidden_size, reuse=tf.get_variable_scope().reuse)
        if keep_prob is not None:
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)
        return cell

    def bidirectional_rnn(inputs, last_relevant, initial_states):
        """
        Bi-directional recurrent network
        """
        with tf.variable_scope('RnnLayer', reuse=reuse):
            initial_states_fw, initial_states_bw = len(hidden_sizes)*[None], len(hidden_sizes)*[None]
            if reuse:
                initial_states_fw, initial_states_bw = initial_states
                initial_states_fw = list(initial_states_fw)
                initial_states_bw = list(initial_states_bw)
            (outputs,
             output_state_fw,
             output_state_bw) = tf.contrib.rnn.stack_bidirectional_dynamic_rnn([rnn_cell(size, keep_prob) for size in hidden_sizes],
                                                                               [rnn_cell(size, keep_prob) for size in hidden_sizes],
                                                                               inputs,
                                                                               initial_states_fw=initial_states_fw,
                                                                               initial_states_bw=initial_states_bw,
                                                                               time_major=False, dtype=tf.float32,
                                                                               sequence_length=last_relevant)
        last_output = batch_gather(outputs, last_relevant) if last_relevant is not None else None
        return outputs, last_output, (output_state_fw, output_state_bw)

    def fw_rnn(inputs, last_relevant, initial_states):
        """
        Forward recurrent network
        """
        with tf.variable_scope('RnnLayer', reuse=reuse):
            stacked_rnn = tf.contrib.rnn.MultiRNNCell([rnn_cell(size, keep_prob) for size in hidden_sizes])
            outputs, states = tf.nn.dynamic_rnn(stacked_rnn, inputs, time_major=False, dtype=tf.float32,
                                                initial_state=initial_states if reuse else None)
            # Gather last relevant
            last_output = batch_gather(outputs, last_relevant) if last_relevant is not None else None
            return outputs, last_output, states

    if direction == 'bidirectional':
        return bidirectional_rnn(inputs, last_relevant, initial_states)
    elif direction == 'forward':
        return fw_rnn(inputs, last_relevant, initial_states)
    else:
        raise Exception('Not implemented.')


def encoder(hidden_units, features, num_samples, keep_prob, direction, reuse_states, batch_size):
    with tf.variable_scope('Encoder'):
        last_embeddings = []; seq_embeddings = []; band_states= []
        for band, band_feature in enumerate(features):
            # Normalize features
            #band_feature = tf.layers.batch_normalization(band_feature, training=is_training, fused=True, reuse=True if band > 0 else None)
            # Process each band independently but using the same weights
            seq_embedding, last_band_embedding, band_state = _rnn_net(band_feature,
                                                                      hidden_sizes=hidden_units,
                                                                      last_relevant=num_samples[band]-1,
                                                                      keep_prob=keep_prob,
                                                                      direction=direction,
                                                                      reuse=True if band > 0 else None,
                                                                      initial_states=band_state if band>0 and reuse_states else None)
            # Append "VARIABLE" LEN (actually fixed because of padding) seq embeedings of bands
            seq_embeddings += [seq_embedding]
            # Append FIXED LEN embeddings of bands
            last_embeddings += [last_band_embedding]

            band_states += [band_state]

        # Stack all band embeddings in a tensor of shape [batch_size, num_units, num_bands]
        last_num_units = hidden_units[-1]*len(features)
        last_num_units = last_num_units*2 if direction=='bidirectional' else last_num_units
        flatten_embeddings = tf.reshape(tf.stack(last_embeddings, axis=-1),
                                        [batch_size, last_num_units])
    return flatten_embeddings, seq_embeddings

def decoder(hidden_units, features, num_samples, keep_prob, direction, reuse_states):
    # RNN  DECODER ----------------------------------------------------------------------------------------
    # Create symmetric decoder
    decoder_reconstruction = []
    with tf.variable_scope('Decoder'):
        for band, band_feature in enumerate(features):
            # Give a "GO" token to the decoder.
            # Note: we might want to fill the encoder with its own feedback rather than with shifted input
            decoder_input = tf.pad(band_feature, [[0, 0], [1, 0], [0, 0]])[:, :-1, :]
            decoded_seq, _, band_state = _rnn_net(decoder_input,
                                                  hidden_sizes=hidden_units,
                                                  last_relevant=None,
                                                  keep_prob=keep_prob,
                                                  direction=direction,
                                                  reuse=True if band > 0 else None,
                                                  initial_states=band_states[band])
            decoder_reconstruction += [decoded_seq]

    return decoder_reconstruction, flatten_embeddings



def autoencoder(features, num_samples, batch_size, seq_len, num_feats, keep_prob, reuse_states=True, direction='forward'):
    """
    Creates recurrent autoencoder 
    """

    # RNN  ENCODER feature extraction layers ----------------------------------------------------------------
    rnn_encoder_hidden_units = [64] + [num_feats]

    if rnn_encoder_hidden_units[-1]!=num_feats:
        raise Exception('Last layer hidden units should be equal to the number of input features to train the autoencoder.')
 
    flatten_embeddings, seq_embeddings = encoder(rnn_encoder_hidden_units, features, num_samples, keep_prob,
            direction, reuse_states, batch_size)

    decoder_reconstruction, flatten_embeddings = decoder(rnn_encoder_hidden_units, seq_embeddings, num_samples, keep_prob,
            direction, reuse_states)

    return decoder_reconstruction, flatten_embeddings

 
def rnn_logits(features, num_samples, is_training, reuse_states=True, direction='forward'):

    keep_prob = 0.85 if is_training else 1.0


    hidden_units = [64,32]
    # Get input shape
    batch_size, seq_len, num_feats = utils.combined_static_and_dynamic_shape(features[0])

    flatten_embeddings, _ = encoder(hidden_units, features, num_samples, keep_prob, direction, reuse_states, batch_size)
    return flatten_embeddings

def fcn_logits(features, is_training):
    activation = 'tanh'
    hidden_units = [512,128,32]
    # Define regularizer
    regularizer = tf.contrib.layers.l2_regularizer(scale=0.05)

    with tf.name_scope('FCN'):
        net = features
        for hidden_unit in hidden_units:
            net = tf.contrib.layers.batch_norm(net, is_training=is_training, fused=True)
            net = tf.layers.dense(net, units=hidden_unit, activation=activation, 
                    kernel_regularizer=regularizer)
    return net

def dense_logits(features,  n_class):
    activation = 'tanh'
    predictions = {}
    with tf.name_scope('Predictions'):
        predictions['scores'] = tf.layers.dense(features, n_class, activation=tf.nn.softmax)
        predictions['classes'] = tf.argmax(predictions['scores'], -1, name='Classes')

    return predictions

"""
"""
import numpy as np
import tensorflow as tf
import decode_tfrecords 

NUM_BANDS = decode_tfrecords.NUM_BANDS

def get_median(v):
    """
    Credit from https://stackoverflow.com/questions/43824665/tensorflow-median-value/43825613#43825613
    """
    mid = tf.shape(v)[0]//2 + 1
    return tf.nn.top_k(v, mid).values[-1]

def _aggregate_bands(dynamic_feats):
    """
    Creates new flux features by aggregating all bands together.
    """
    # Concat all band fluxes and times
    flux_list = tf.concat([dynamic_feats['augmented_flux_%i'%band] for band in range(NUM_BANDS)], axis=0)
    time_list = tf.concat([dynamic_feats['time_diff_%i'%band] for band in range(NUM_BANDS)], axis=0)

    # Sort time values
    args_ordered = tf.contrib.framework.argsort(time_list)

    # Create one_hot encodings of band fluxes
    band_features = tf.concat([tf.fill(tf.shape(dynamic_feats['augmented_flux_%i'%band]), band) for band in range(NUM_BANDS)], axis=0)
    band_features = tf.gather(band_features, args_ordered)
    band_features = tf.one_hot(band_features, NUM_BANDS, on_value=1., off_value=0.0)

    # Gather sroted (by time) fluxes
    agg_flux  = tf.gather(tf.concat(flux_list, axis=0), args_ordered)

    # Create new a feature that contains:
    # [one_hot_band_idx[num_samples, 6], agg_fluxes[num_samples], agg_times[num_samples]]
    agg_features = tf.concat([band_features, tf.expand_dims(agg_flux, axis=-1),\
            tf.expand_dims(tf.gather(time_list, args_ordered), axis=-1)], axis=-1)
    return agg_features, agg_flux, tf.gather(time_list, args_ordered)

def _preprocess_ndft(dfts, train_stats):
    """
    Normalize NDFT features with train statistics if provided.
    """
    # Split dfts in magnitude and phase list of tensors
    mag_dfts   = [dfts[band][:, 0] for band in range(NUM_BANDS)]
    phase_dfts = [dfts[band][:, 1] for band in range(NUM_BANDS)]

    # Standardize by train statistics if they're provided
    if train_stats is not None:
        preprocessed_dfts = []
        for band in range(NUM_BANDS):
            mag_dft, phase_dft = mag_dfts[band], phase_dfts[band]
            # Standardize dft
            band_stats = train_stats['dft_%i'%band]
            mag_dft = (mag_dft - band_stats['mag/mean'])/(band_stats['mag/std'] + 1e-12)
            phase_dft = (phase_dft - band_stats['phase/mean'])/(band_stats['phase/std'] + 1e-12)
            preprocessed_band = tf.stack([mag_dft, phase_dft], axis=-1)
            preprocessed_dfts += [preprocessed_band]
    # Otherwise scale all bands together
    else:
        mag_dfts = _max_min_normalize(mag_dfts)
        phase_dfts = _max_min_normalize(phase_dfts)
        preprocessed_dfts = [tf.stack([mag, phase], axis=-1) for mag, phase in zip(mag_dfts, phase_dfts)]
    return preprocessed_dfts

def _max_min_normalize(tensors):
    """
    Max/min normalize a tensors, optionally a list of tensor together.
        1. Compute overall min and max of all tensors
        2. Normalize each tensor with common stats
    """
    if isinstance(tensors, list):
        concated_tensors = tf.concat(tensors, axis=0)
        min = tf.reduce_min(concated_tensors, axis=0)
        max = tf.reduce_max(concated_tensors, axis=0)
        return [(tensor - min)/(max - min) for tensor in tensors]
    else:
        min = tf.reduce_min(tensors, axis=0)
        max = tf.reduce_max(tensors, axis=0)
        return (tensors - min)/(max - min)

def _standard_normalize(tensors, std_epsilon=1e-12):
    """
    Normalize a list of tensors by its global statistical moments.
        1. Compute overall mean and std of all tensors
        2. Normalize each tensor with global stats
    """
    # Compute aggregated statistics for all bands
    concated_tensors = tf.concat(tensors, axis=0)
    mean, std = tf.nn.moments(concated_tensors, axes=0)
    # Normalize each band flux
    return  [(tensor - mean )/(std + std_epsilon) for tensor in tensors]

def _add_gaussian_noise(tensor, std):
    """
    Sample noise from a multivariate normal distribution, where std is a tensor of rank N.
    """
    # Create a multivariate normal distribution, zero-mean and std=flux_err=[3, 4.5, 6...]
    multivariate_normal_dist = tf.distributions.Normal(loc=0.,
                                                       scale=std,
                                                       validate_args=False, # performance degradation
                                                       name='GaussianNoiseGenerator')
    # Generates a single noise sample
    noise = multivariate_normal_dist.sample()
    return tensor + noise

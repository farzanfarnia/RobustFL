import tflib as lib

import numpy as np
import tensorflow as tf
from numpy import linalg as LA
import warnings

_default_weightnorm = False
def enable_default_weightnorm():
    global _default_weightnorm
    _default_weightnorm = True

_weights_stdev = None
def set_weights_stdev(weights_stdev):
    global _weights_stdev
    _weights_stdev = weights_stdev

def unset_weights_stdev():
    global _weights_stdev
    _weights_stdev = None

def l2_norm(input_x, epsilon=1e-12):
    """normalize input to unit norm"""
    input_x_norm = input_x/(tf.reduce_sum(input_x**2)**0.5 + epsilon)
    return input_x_norm


def Conv2D(name, input_dim, output_dim, filter_size, inputs, he_init=True, spectral_norm=False, tighter_sn=False, beta=1. , update_collection=None, reuse =None, mask_type=None, stride=1, weightnorm=None, biases=True, gain=1.):
    """
    inputs: tensor of shape (batch size, num channels, height, width)
    mask_type: one of None, 'a', 'b'

    returns: tensor of shape (batch size, num channels, height, width)
    """
    with tf.name_scope(name) as scope:

        inputs_shape = inputs.get_shape().as_list()
        
        if mask_type is not None:
            mask_type, mask_n_channels = mask_type

            mask = np.ones(
                (filter_size, filter_size, input_dim, output_dim), 
                dtype='float32'
            )
            center = filter_size // 2

            # Mask out future locations
            # filter shape is (height, width, input channels, output channels)
            mask[center+1:, :, :, :] = 0.
            mask[center, center+1:, :, :] = 0.

            # Mask out future channels
            for i in xrange(mask_n_channels):
                for j in xrange(mask_n_channels):
                    if (mask_type=='a' and i >= j) or (mask_type=='b' and i > j):
                        mask[
                            center,
                            center,
                            i::mask_n_channels,
                            j::mask_n_channels
                        ] = 0.


        def uniform(stdev, size):
            return np.random.uniform(
                low=-stdev * np.sqrt(3),
                high=stdev * np.sqrt(3),
                size=size
            ).astype('float32')

        fan_in = input_dim * filter_size**2
        fan_out = output_dim * filter_size**2 / (stride**2)

        if mask_type is not None: # only approximately correct
            fan_in /= 2.
            fan_out /= 2.

        if he_init:
            filters_stdev = np.sqrt(4./(fan_in+fan_out))
        else: # Normalized init (Glorot & Bengio)
            filters_stdev = np.sqrt(2./(fan_in+fan_out))

        if _weights_stdev is not None:
            filter_values = uniform(
                _weights_stdev,
                (filter_size, filter_size, input_dim, output_dim)
            )
        else:
            filter_values = uniform(
                filters_stdev,
                (filter_size, filter_size, input_dim, output_dim)
            )

        # print "WARNING IGNORING GAIN"
        filter_values *= gain
        
        filters = lib.param(name+'.Filters', filter_values)
        
        
        if weightnorm==None:
            weightnorm = _default_weightnorm
        if weightnorm:
            norm_values = np.sqrt(np.sum(np.square(filter_values), axis=(0,1,2)))
            target_norms = lib.param(
                name + '.g',
                norm_values
            )
            with tf.name_scope('weightnorm') as scope:
                norms = tf.sqrt(tf.reduce_sum(tf.square(filters), reduction_indices=[0,1,2]))
                filters = filters * (target_norms / norms)
        
        if mask_type is not None:
            with tf.name_scope('filter_mask'):
                filters = filters * mask
        
        #if spectral_norm:
        #    filters = weights_spectral_norm(filters, update_collection=update_collection, name = name + '.Filters', reuse=reuse)
            
        if spectral_norm:
            filters = weights_spectral_norm(filters, update_collection=update_collection,
                                            name = name + '.Filters', reuse=reuse,
                                            tighter_sn=tighter_sn, u_width=inputs_shape[-2], beta=beta,
                                            u_depth=inputs_shape[-3], stride=stride, padding='SAME')

        
        result = tf.nn.conv2d(
            input=inputs, 
            filter=filters, 
            strides=[1, 1, stride, stride],
            padding='SAME',
            data_format='NCHW'
        )

        if biases:
            _biases = lib.param(
                name+'.Biases',
                np.zeros(output_dim, dtype='float32')
            )

            result = tf.nn.bias_add(result, _biases, data_format='NCHW')

        return result

def weights_spectral_norm(weights, u=None, Ip=1, update_collection=None,
                          reuse=False, name='weights_SN', beta=1.,
                          tighter_sn=False, u_width=28, u_depth=3, stride=1, padding='SAME'):
    """Perform spectral normalization"""

    def power_iteration(u, w_mat, Ip):
        u_ = u
        for _ in range(Ip):
            v_ = l2_norm(tf.matmul(u_, tf.transpose(w_mat)))
            u_ = l2_norm(tf.matmul(v_, w_mat))
        return u_, v_

    def power_iteration_conv(u, w_mat, Ip):
        u_ = u
        for _ in range(Ip):
            v_ = l2_norm(tf.nn.conv2d(u_, w_mat, strides=[1, stride, stride, 1], padding=padding))
            u_ = l2_norm(tf.nn.conv2d_transpose(v_, w_mat, [1, u_width, u_width, u_depth],
                                                strides=[1, stride, stride, 1], padding=padding))
        return u_, v_

    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        w_shape = weights.get_shape().as_list()
        
        # The tighter spectral normalization approach breaks the [f_in, f_out, d_in, d_out] filters
        # into a set of f_in*f_out subfilters each of size d_in*d_out.
        # ONLY USE THIS FOR conv2d LAYERS. Original sn works better for fully-connected layers
        if tighter_sn:
            if u is None:
                # Initialize u (our "eigenimage")
                u = tf.get_variable('u', shape=[1, u_width, u_width, u_depth], 
                                    initializer=tf.truncated_normal_initializer(), trainable=False)

            u_hat, v_hat = power_iteration_conv(u, weights, Ip)
            z = tf.nn.conv2d(u_hat, weights, strides=[1, stride, stride, 1], padding=padding)
            sigma = tf.maximum(tf.reduce_sum(tf.multiply(z, v_hat))/beta, 1)
            
            if update_collection is None:
                with tf.control_dependencies([u.assign(u_hat)]):
                    w_norm = weights/sigma
            else:
                tf.add_to_collection(update_collection, u.assign(u_hat))
                w_norm = weights/sigma

        # Use the spectral normalization proposed in SN-GAN paper
        else:
            if u is None:
                u = tf.get_variable('u', shape=[1, w_shape[-1]], 
                                    initializer=tf.truncated_normal_initializer(), trainable=False)

            w_mat = tf.reshape(weights, [-1, w_shape[-1]])
            u_hat, v_hat = power_iteration(u, w_mat, Ip)
            sigma = tf.maximum(tf.matmul(tf.matmul(v_hat, w_mat), tf.transpose(u_hat))/beta, 1)
            
            w_mat = w_mat/sigma

            if update_collection is None:
                with tf.control_dependencies([u.assign(u_hat)]):
                    w_norm = tf.reshape(w_mat, w_shape)
            else:
                tf.add_to_collection(update_collection, u.assign(u_hat))
                w_norm = tf.reshape(w_mat, w_shape)

        tf.add_to_collection('w_after_sn', w_norm)

        return w_norm
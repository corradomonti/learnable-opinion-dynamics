""" The Learnable Opinion Dynamics generative model, implemented in Tensorflow. """

import numpy as np
import tensorflow as tf

from collections import namedtuple

BATCHSIZE = None
EPSILON = 1E-04

def map_gather_nd(tensor, indexes, name=None):
    """ Returns a tensor composed by the `indexes[0]`-th element of `tensor[0]`,
        the `indexes[1]`-th element of `tensor[1]`, and so on.
        
        The resulting tensor will have one dimension less than the original `tensor`.
    """
    assert len(indexes.shape) == 1, "indexes must be a 1-dimensional vector."
    range_index_pairs = tf.stack([tf.range(tf.shape(indexes)[0]), indexes], axis=1)
    return tf.gather_nd(tensor, range_index_pairs, name=name)
    

def beta_kl_divergence(sample, prior_alpha, prior_beta):
    mu = tf.math.reduce_mean(sample)
    var = tf.reduce_mean(tf.squared_difference(sample, mu)) + EPSILON
    observed_alpha = ((1. - mu) / var - (1. / (mu + EPSILON))) * tf.square(mu)
    observed_beta = observed_alpha * (1. / (mu + EPSILON) - 1)
    return (
        tf.lgamma(prior_alpha + prior_beta)
        - (tf.lgamma(prior_alpha) + tf.lgamma(prior_beta))
        - (tf.lgamma(observed_alpha + observed_beta + EPSILON))
        + (tf.lgamma(observed_alpha + EPSILON) + tf.lgamma(observed_beta + EPSILON))
        + (prior_alpha - observed_alpha) * (tf.digamma(prior_alpha) - tf.digamma(prior_alpha + prior_beta))
        + (prior_beta - observed_beta) * (tf.digamma(prior_beta) - tf.digamma(prior_alpha + prior_beta))
    )

def build_update_matrix(adjacency_values, edge_signs, mu_pos, mu_neg):
    """ Returns a tensorflow vector such that i-th is (-mu_neg * adjacency_values[i]) if 
    edge_signs[i] < 0 or (mu_pos * adjacency_values[i]) if edge_signs[i] > 0. """
    return (
        adjacency_values * tf.minimum(0., edge_signs * mu_neg) + 
        adjacency_values * tf.maximum(0., edge_signs * mu_pos)
    )

def build_opinions_matrix(N, x_0, adjacency_indices_iterator, adjacency_values_iterator, signs_iterator, mu_pos, mu_neg):
    x_t = x_0
    x_vectors = [x_t]
    summaries = [tf.summary.histogram(f"x_0", x_0)]
    
    data_iterator = enumerate(zip(adjacency_indices_iterator, adjacency_values_iterator, signs_iterator))
    for t, (adjacency_indices, adjacency_values_numerical, signs_numerical) in data_iterator:
        assert len(adjacency_indices) == len(adjacency_values_numerical) == len(signs_numerical), "Inconsistent sparse representation."
        if len(adjacency_indices) == 0: # If the graph at t is emtpy:
            x_t_plus_one = tf.identity(x_t, name=f"x{t+1}")
            x_vectors.append(x_t_plus_one)
            x_t = x_t_plus_one
            continue
            
        edge_signs = tf.constant(signs_numerical, dtype=tf.float32, name=f'sign_{t}')
        adjacency_values = tf.constant(adjacency_values_numerical, dtype=tf.float32, name=f'adj_values_{t}')
        adjacency_graph = tf.SparseTensor(indices=adjacency_indices,
                                      values=build_update_matrix(adjacency_values, edge_signs, mu_pos, mu_neg),
                                      dense_shape=[N, N])
                                      
        coefficients = 1 - tf.sparse_reduce_sum(adjacency_graph, axis=0)
        coefficients.set_shape(N) # This is not necessary in tensorflow >= 1.13, since it can be inferred.
        retained_opinion = tf.multiply(coefficients, x_t)
        updating_opinion = tf.squeeze(tf.sparse.matmul(
            tf.sparse.transpose(adjacency_graph), tf.expand_dims(x_t, axis=1)))
        x_t_plus_one = tf.clip_by_value(updating_opinion + retained_opinion, -1, 1, name=f"x{t+1}")
        x_t_plus_one.set_shape(N)
        x_vectors.append(x_t_plus_one)

        x_t = x_t_plus_one
        summaries += [
            tf.summary.histogram(f"x_{t+1}", x_t_plus_one)
        ]
	
    return tf.stack(x_vectors, name='X'), x_t, summaries

def build_alpha_t(x_t, N, alpha_clip, consensus_threshold, backfire_threshold):
    tiled = tf.tile([x_t], [N, 1])
    diff = tf.abs(tiled - tf.transpose(tiled))
    frac_positive = tf.reduce_mean(tf.cast(diff < consensus_threshold, tf.float32))
    frac_negative = tf.reduce_mean(tf.cast(diff > backfire_threshold, tf.float32))
    alpha_t = frac_positive / (frac_positive + frac_negative)
    return tf.clip_by_value(alpha_t, -1. + alpha_clip, 1 - alpha_clip, name="alpha_t")

def build_tf_model(N, Q, adjacency_indices_iterator, adjacency_values_iterator, signs_iterator,
            active_nodes_vector,
            consensus_threshold=0.6, backfire_threshold=1.2,
            link_sharpness=16, feature_sharpness=8, mu_pos=0.001, mu_neg=0.001,
            alpha_clip=0.05, sigma_prior_p=8., sigma_prior_q=8., initial_range=0.001,
            fixed_features=None):
    """
    Build the Tensorflow graph of operations that represents the likelihood of the model.
    
    Args:
        N (int): number of nodes in the graph.
        Q (int): number of features.
        adjacency_indices_iterator, adjacency_values_iterator, signs_iterator: sparse matrix representation of the graph before t.
        active_nodes_vector: numpy vector of active nodes (1 where i is active, 0 where is not).
        consensus_threshold: parameter controlling the maximum distance where positive interactions are likely.
        backfire_threshold:  parameter controlling the minimum distance where negative interactions are likely.
        fixed_features (dict or None): dictionary of {feature index: feature value} to fix W_a for some features.
    """
    tf.reset_default_graph()
                                          
    with tf.name_scope('hyperparameters'):
        mu_pos = tf.constant(mu_pos, name='mu_pos')
        mu_neg = tf.constant(mu_neg, name='mu_neg')

    
    with tf.name_scope('opinions'):
        opinion_constraint = lambda x: tf.clip_by_value(x, -1, 1)
        opinion_initializer = tf.initializers.random_uniform(-initial_range, initial_range)
        
        x_0 = tf.get_variable('x_0', shape=(N, ), initializer=opinion_initializer)
        X, x_t, opinion_summaries = build_opinions_matrix(N,
                                        opinion_constraint(x_0), # For technical reason, the x_0 variable can't be constrained, so we'll use its clipped value as x_0.
                                        adjacency_indices_iterator,
                                        adjacency_values_iterator,
                                        signs_iterator,
                                        mu_pos=mu_pos, mu_neg=mu_neg)
        
        alpha = build_alpha_t(x_t, N, alpha_clip, consensus_threshold, backfire_threshold)

    common_summary = opinion_summaries + [
        tf.summary.scalar('alpha', alpha),
        tf.summary.scalar('mu_pos', mu_pos),
        tf.summary.scalar('mu_neg', mu_neg)
    ]

    free_w = tf.get_variable('w', shape=(Q, ), initializer=opinion_initializer, constraint=opinion_constraint)
    if fixed_features is not None:
        fixed_values = np.zeros(Q, dtype=np.float32)
        is_fixed = np.zeros(Q, dtype=np.bool)
        for fixed_index, fixed_value in fixed_features.items():
            fixed_values[fixed_index] = fixed_value
            is_fixed[fixed_index] = True 
        w = ~is_fixed * free_w + is_fixed * fixed_values
    else:
        w = free_w
        
    sigma = tf.get_variable('sigma', shape=(Q, ), initializer=tf.constant_initializer(0.5),
                        constraint=lambda x: tf.clip_by_value(x, 1E-2, 1 - 1E-2))
    sigma_prior_loss = beta_kl_divergence(sigma, sigma_prior_p, sigma_prior_q)
    # sigma = tf.constant(np.full(Q, 0.5, dtype=np.float32), name='sigma')
                                          
    # ### Optimizing node feature links

    node = tf.placeholder(tf.int32, shape=(BATCHSIZE, ), name='node')
    feature = tf.placeholder(tf.int32, shape=(BATCHSIZE, ), name='a')
    
    x_t_node = tf.gather_nd(x_t, tf.expand_dims(node, 1), name='x_t_node')
    node_all_features_distance = tf.abs(tf.transpose(tf.ones((Q, 1)) * x_t_node) - w)
    node_all_features_likelihood = tf.sigmoid(-feature_sharpness * (node_all_features_distance - sigma))
    kappa_node_feature = map_gather_nd(node_all_features_likelihood, feature) # K_a(x_{t, v})
    normalize_node_feature = tf.math.reduce_sum(node_all_features_likelihood, axis=1) # \sum_{a'} K_{a'}(x_{t, v})
    likelihood_node_feature = tf.divide(kappa_node_feature, normalize_node_feature + EPSILON)

    feature_loss = tf.losses.compute_weighted_loss(- tf.math.log(likelihood_node_feature + EPSILON))

    node_feature_summaries = tf.summary.merge(common_summary + [
        tf.summary.histogram("w", w),
        tf.summary.histogram("sigma", sigma),
        tf.summary.histogram("kappa_node_feature", kappa_node_feature),
        tf.summary.histogram("normalize_node_feature", normalize_node_feature),
        tf.summary.histogram("likelihood_node_feature", likelihood_node_feature),
        tf.summary.histogram("feature_loss", feature_loss)
    ])

    # ### Optimizing node node links
    
    u = tf.placeholder(tf.int32, shape=(BATCHSIZE, ), name='u')
    v = tf.placeholder(tf.int32, shape=(BATCHSIZE, ), name='v')
    q_pos_fixed = tf.placeholder(tf.float32, shape=(BATCHSIZE, ), name='q_pos_fixed')
    q_neg_fixed = tf.placeholder(tf.float32, shape=(BATCHSIZE, ), name='q_neg_fixed')
    active_nodes = tf.constant(active_nodes_vector, dtype=tf.float32)

    x_tu = tf.gather_nd(x_t, tf.expand_dims(u, 1), name='x_tu')
    
    node_node_distances = tf.abs(tf.transpose(tf.ones((N, 1)) * x_tu) - x_t)

    kappa_pos_all = tf.sigmoid(-link_sharpness * (node_node_distances - consensus_threshold))
    kappa_neg_all = tf.sigmoid(link_sharpness * (node_node_distances - backfire_threshold))
    
    kappa_pos_v = map_gather_nd(kappa_pos_all, v, name='kappa_pos_v')
    kappa_pos_sum =tf.einsum('iv,v->i', kappa_pos_all, active_nodes, name='kappa_pos_sum')
    pos_likelihood = tf.divide(kappa_pos_v + EPSILON, kappa_pos_sum + EPSILON, name='pos_likelihood')
    
    kappa_neg_v = map_gather_nd(kappa_neg_all, v, name='kappa_neg_v')
    kappa_neg_sum = tf.einsum('iv,v->i', kappa_neg_all, active_nodes, name='kappa_neg_sum')
    neg_likelihood = tf.divide(kappa_neg_v + EPSILON, kappa_neg_sum + EPSILON, name='neg_likelihood')
        
    likelihood_link = tf.add(
        q_pos_fixed * pos_likelihood,
        q_neg_fixed * neg_likelihood,
        name='link_likelihood')
    link_loss = tf.losses.compute_weighted_loss(- tf.math.log(likelihood_link + EPSILON))
    
    q_pos = tf.divide(alpha * pos_likelihood, pos_likelihood + neg_likelihood + EPSILON)
    q_neg = tf.divide((1. - alpha) * neg_likelihood, pos_likelihood + neg_likelihood + EPSILON)

    node_node_summaries = tf.summary.merge(
            common_summary + [
                tf.summary.histogram("kappa_pos_v", kappa_pos_v),
                tf.summary.histogram("kappa_pos_sum", kappa_pos_sum),
                tf.summary.histogram("kappa_neg_v", kappa_neg_v),
                tf.summary.histogram("kappa_neg_sum", kappa_neg_sum),
                tf.summary.histogram("pos_likelihood", pos_likelihood),
                tf.summary.histogram("neg_likelihood", neg_likelihood),
                tf.summary.histogram("likelihood_link", likelihood_link),
                tf.summary.histogram("link_loss", link_loss)
            ]
    )
        
    return namedtuple("TfModel", """
        alpha, mu_neg, mu_pos, node, feature, X,
        q_neg, q_neg_fixed, q_pos, q_pos_fixed, sigma, sigma_prior_loss,
        kappa_neg_all, kappa_node_feature, kappa_pos_all,
        likelihood_link, likelihood_node_feature, feature_loss, link_loss,
        pos_likelihood, neg_likelihood, node_feature_summaries, node_node_summaries,
        normalize_node_feature, u, v, w, free_w, x_0, x_t
    """)(
        alpha, mu_neg, mu_pos, node, feature, X,
        q_neg, q_neg_fixed, q_pos, q_pos_fixed, sigma, sigma_prior_loss,
        kappa_neg_all, kappa_node_feature, kappa_pos_all,
        likelihood_link, likelihood_node_feature, feature_loss, link_loss,
        pos_likelihood, neg_likelihood, node_feature_summaries, node_node_summaries,
        normalize_node_feature, u, v, w, free_w, x_0, x_t
    )

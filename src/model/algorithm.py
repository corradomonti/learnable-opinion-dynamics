""" The inference algorithm for learnable opinion dynamics. """

from .evaluate import evaluate_model
from .graph import Graph
from .tfmodel import build_tf_model

import numpy as np
import tensorflow as tf

from collections import namedtuple

ModelResults = namedtuple("ModelResults", "X, w, sigma, t2signs, alphas, evals")

class ConvergenceChecker:
    """ Class to to check if a set of variables reach convergence. """
    def __init__(self, threshold, verbose=False):
        self.last_values = None
        self.threshold = threshold
        self.verbose = verbose
        self.iteration = 0
        
    def is_converged(self, **values):
        """ Returns True if the mean of the differences w.r.t. the last call of this
            method is less than the threshold for each of the given values.
        """
        if self.last_values is None:
            self.last_values = values
            return False
        
        assert set(self.last_values.keys()) == set(values.keys())
        diffs = {k: np.mean(np.abs(self.last_values[k] - values[k]).flatten()) for k in values}
        if self.verbose:
            stats = '\t'.join("%s: %.01E" % (k, diffs[k]) for k in sorted(diffs))
            print("Iteration %d.\tDiffs:\t%s" % (self.iteration, stats))
        self.last_values = values
        self.iteration += 1
        return all(v < self.threshold for v in diffs.values())
        
def em_step(graph, node_features, signs_until_t, Q, t, x_0=None, w=None, sigma=None,
        learning_rate_link=0.0001, learning_rate_feature=0.001, threshold=5E-4, max_iter=1000,
        sigma_prior_coefficient=1, verbose=True,
        **hyperparameters):
    """ EM step to find (or refine) opinions and signs for **a single** time step.

    Args:
        graph (Graph): The interaction graph, as an object of class `Graph` defined in `graph.py`.
        node_features (list): node-features arcs of this time step, as a list of (v, a) arcs.
        signs_until_t (list): the signs of the previous time steps, as a list of list of numbers;
            each list is one time step, and inside that list the sign of each arc appear in the same
            order as the list of arcs given by `graph.get_interaction_graph(t)`.
        Q (int): Number of features.
        t (int): This time step.
        
        x_0 (np.array, optional): Initial value for x_0.
        w (np.array, optional): Initial value for w.
        learning_rate_link (float, optional): learning rate for node-node arcs.
        learning_rate_feature (float, optional): learning rate for node-feature arcs.
        threshold (float, optional): threshold for convergence.
        max_iter (int, optional): maximum number of iteration, if no convergence is detected.
        **hyperparameters: other keyword arguments are passed to `build_tf_model(..)` as hyperparameters.

    Returns:
        Tuple:
        - x_0 (numpy array): Array of length N, in x_0[u] has the opinion of node u at time step 0.
        - w (numpy array): Array of length Q representing in w[a] the opinion of feature a.
        - sigma (numpy array): Array of length Q representing in w[a] the acceptance width of feature a.
        - X (numpy matrix): Matrix of size N x (t + 1) representing all the opinions over time.
        - signs (list): list containing the signs of each arc in this time step, in the same order
          as the list of arcs given by `graph.get_interaction_graph(t)`.
        - num_iterations (int): number of iterations before convergence.
        - evaluation (dict): evaluation metrics: we run in-sample evaluation to measure the
          training error.
    
    """    
    adj_indices, adj_values = graph.get_interaction_graph(t)
    
    u, v = np.array(graph.adj_indices[t]).T if len(graph.adj_indices[t]) > 0 else (np.array([]), np.array([]))
    node, feature = np.array(node_features).T if len(node_features) > 0 else (np.array([]), np.array([]))
        
    
    model = build_tf_model(graph.N, Q, adj_indices, adj_values, signs_until_t,
                      graph.get_active_nodes(t), **hyperparameters)
    
    link_optimizer = tf.train.AdamOptimizer(learning_rate_link).minimize(model.link_loss)
    feature_optimizer = tf.train.AdamOptimizer(learning_rate_feature).minimize(
        model.feature_loss + sigma_prior_coefficient * model.sigma_prior_loss)
            
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        if x_0 is not None: model.x_0.load(x_0)
        if w is not None: model.free_w.load(w)
        if sigma is not None: model.sigma.load(sigma)
        
        check = ConvergenceChecker(verbose=verbose, threshold=threshold)
        while True:
            q_pos, q_neg = sess.run([model.q_pos, model.q_neg], feed_dict={
                     model.u: u,
                     model.v: v,
                    })

            _, _summaries = sess.run([link_optimizer, model.node_node_summaries], feed_dict={
                            model.u: u,
                            model.v: v,
                            model.q_pos_fixed: q_pos,
                            model.q_neg_fixed: q_neg,
                        })
            
            _, _summaries = sess.run([feature_optimizer, model.node_feature_summaries], feed_dict={
                    model.node: node,
                    model.feature: feature,
                })
                
            x_0, w, sigma = sess.run([model.x_0, model.w, model.sigma])
            
            if check.is_converged(x_0=x_0, w=w, q_pos=q_pos, q_neg=q_neg, sigma=sigma) or check.iteration > max_iter:
                signs = (q_pos / (q_pos + q_neg)) * 2 - 1
                eval_results = evaluate_model(model, list(zip(u, v)), node_features, graph.N, Q)
                return x_0, w, sigma, model.X.eval(), signs, model.alpha.eval(), check.iteration, eval_results
        
def learn_opinion_dynamics(N, Q, T, u_v_t_weights, v_a_t_weights, num_epochs=2, num_restarts=1,
    alpha_clip_0=None, **hyperparameters):
    """
    Iterative EM algorithm to find opinions and signs, given a temporal interaction graph and
    a temporal bipartite node-feature graph.

    Args:
        N (int): Number of nodes.
        Q (int): Number of features.
        T (int): Number of time steps.
        u_v_t_weights (list): The interaction graph, as a list of (u, v, t, w) tuples meaning that
                 node u interacts with node v with weight w at time t. Those do not need to be
                 sorted and can contain duplicates (duplicates will be unified and their weight
                 will be summed). The semantic of u -> v is "u tries to change v's opinion".
                 All nodes must be positive integers lower than N. 
        
        v_a_t_weights (list): The node-feature graph, as a list of (v, a, t, w) tuples meaning that
                 node v has feature a with weight w at time t. Those do not need to be
                 sorted and can contain duplicates (duplicates will be unified and their weight
                 will be summed). All features must be positive integers lower than Q.
        
        num_epochs (int, optional): The number of epochs for training.
        num_restarts (int, optional): Number of multiple restarts.
        alpha_clip_0 (int, optional): Change the clipping value for alpha_t only for t=0 in the first epoch.
        **hyperparameters: all other keyword arguments are passed to `em_step(..)` as hyperparameters.

    Returns:
        Tuple:
        - X (numpy matrix): Matrix N x T representing the evolution of opinions. X[u, t] is the
                 opinion of u at time t.
        - w (numpy array): Array of length Q representing in w[a] the opinion of feature a.
        - sigma (numpy array): Array of length Q representing in w[a] the acceptance width of feature a.
        - signs (list): list containing the signs of each arc. This is a list where each element
                 corresponds to a specific time step; the t-th element of `signs` is composed by
                 tuples ((u, v), s) meaning that the arc u -> v at time t has sign s.
        - evaluations (list): list of T dictionaries, where each one is the evaluation metrics at t for the last epoch.
    """
    interactions = Graph(u_v_t_weights, T=T, N=N)
    results = []
    
    for restart in range(num_restarts):
        print(f">>>>>> Restart {restart+1}/{num_restarts} <<<<<")
        x_0 = None
        w = None
        sigma = None
        alphas = np.full(T, np.nan)
        hyperparameters_0 = hyperparameters.copy()
        if alpha_clip_0 is not None:
            hyperparameters_0['alpha_clip'] = alpha_clip_0
            
        for epoch in range(num_epochs):
            print(f"################## EPOCH {epoch} ##################")
            last_epoch_evaluations = []
            signs_until_t = []
            total_iterations = 0
            for t in range(T):
                print(f".................. time step {t} ...............")
                node_features = [(v, a) for (v, a, _t, _) in v_a_t_weights if _t == t]
                x_0, w, sigma, X, signs_t, alpha, num_iterations, evaluation = em_step(
                    interactions, node_features, signs_until_t,
                    Q=Q, t=t, x_0=x_0, w=w, sigma=sigma,
                    **(hyperparameters_0 if t == 0 and epoch == 0 else hyperparameters))
                print(num_iterations, "iterations")
                last_epoch_evaluations.append(evaluation)
                signs_until_t.append(np.sign(signs_t))
                total_iterations += num_iterations
                alphas[t] = alpha
            t2signs = [list(zip(interactions.adj_indices[t], signs_until_t[t])) for t in range(T)]
            if total_iterations == T: # Meaning 1 (the minimum) for each time step.
                print("Early stop.")
                break
        
        aggregated_evals = {key: np.mean([t_eval[key] for t_eval in last_epoch_evaluations])
                            for key in last_epoch_evaluations[0]}
        
        results.append(ModelResults(X, w, sigma, t2signs, alphas, aggregated_evals))
    
    return min(results, key=lambda x: x.evals['log_loss'])
        

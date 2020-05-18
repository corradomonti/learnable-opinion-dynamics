""" Internal data representation formats for temporal graphs. """

import numpy as np

from collections import defaultdict

class Graph:
    """ Represents the full temporal interaction graph in many convenient formats. """
    
    def __init__(self, u_v_t_weights, T, N):
        """ Represents the full temporal interaction graph in many convenient formats.
        
            Args:
                u_v_t_weights (list): The interaction graph, as a list of (u, v, t, w) tuples
                    meaning that node u interacts with node v with weight w at time t. Those do not
                    need to be sorted and can contain duplicates (duplicates will be unified and
                    their weight will be summed). The semantic of u -> v is "u tries to change
                    v's opinion".
                T (int): Number of time steps.
                N (int): Number of nodes.
        """
        self.u_v_t_weights = u_v_t_weights
        self.T = T
        self.N = N
        assert all(u < N and v < N and t < T and w >= 0 for u, v, t, w in u_v_t_weights)
        self.active_node_matrix = self._build_active_node_matrix(u_v_t_weights, T, N)
        self.adj_indices, self.adj_values = self._build_interaction_graph(u_v_t_weights, T)
    
    @staticmethod
    def _build_active_node_matrix(u_v_t_weights, T, N):
        t2active_nodes_set = [set() for _ in range(T)]
        for _u, v, t, _w in u_v_t_weights:
            t2active_nodes_set[t].add(v)

        active_node_matrix = np.zeros(shape=(T, N))
        for t in range(T):
            active_node_matrix[t, list(t2active_nodes_set[t])] = 1
            
        return active_node_matrix
    
    @staticmethod
    def _build_interaction_graph(u_v_t_weights, T):
        t2uv2weight = [defaultdict(int) for _ in range(T)]
        for u, v, t, weight in u_v_t_weights:
            t2uv2weight[t][(u, v)] += weight
        
        adj_indices = []
        adj_values = []
        for t in range(T):
            adj_indices_at_t = []
            adj_values_at_t = []
            for (u, v), weight in t2uv2weight[t].items():
                adj_indices_at_t.append((u, v))
                adj_values_at_t.append(weight)
            adj_indices.append(adj_indices_at_t)
            adj_values.append(adj_values_at_t)
            
        return adj_indices, adj_values
            
    def get_active_nodes(self, t):
        """ Returns the N-length vector that is 1 where i-th is active at the given time step. """
        assert t < self.T
        return self.active_node_matrix[t]
        
    def get_interaction_graph(self, t):
        """ Returns sparse representations of the graph in the time steps before the given one. """
        assert t <= self.T
        return self.adj_indices[:t], self.adj_values[:t]
    
    

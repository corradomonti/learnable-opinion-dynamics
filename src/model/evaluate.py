import numpy as np
import sklearn.metrics

from abc import ABC, abstractmethod
from collections import defaultdict

METRICS = [sklearn.metrics.roc_auc_score, sklearn.metrics.log_loss, sklearn.metrics.average_precision_score]

class BaselinePredictor(ABC):
    @abstractmethod
    def score(self, node, feature):
        raise NotImplementedError()
        
class AveragePredictor(BaselinePredictor):
    def __init__(self, v_a_t_weights, use_weight=False, only_timestep=None):
        self.va2count = defaultdict(int)
        for v, a, t, w in v_a_t_weights:
            if only_timestep is None or t == only_timestep:
                self.va2count[(v, a)] += (w if use_weight else 1)
        self.max_count = max(self.va2count.values())
        
    def score(self, node, feature):
        return self.va2count[(node, feature)] / self.max_count
        
def get_baselines(v_a_t_weights):
    return {'baseline': AveragePredictor(v_a_t_weights)}

def _build_links_sample(link_sequence, N, Q=None):
    existing_links = set(link_sequence)
    if Q is None:
        active_nodes = np.array(list(set(j for _i, j in link_sequence)))
    
    head = []
    tail = []
    is_link = []

    for i, j in link_sequence:
        head.append(i)
        tail.append(j)
        is_link.append(1)

        while True:
            i = np.random.randint(N)
            j = np.random.choice(active_nodes) if Q is None else np.random.randint(Q)
            if (i, j) not in existing_links:
                head.append(i)
                tail.append(j)
                is_link.append(0)
                break
    return np.array(head), np.array(tail), np.array(is_link)

def evaluate_model(model, node_node_links, node_feature_links, N, Q, baselines=None):
    alpha = model.alpha.eval()
    node, feat, is_nf_link = _build_links_sample(node_feature_links, N, Q)
    u, v, is_nn_link = _build_links_sample(node_node_links, N)
    nf_likelihood = model.likelihood_node_feature.eval(feed_dict={
        model.node: node, model.feature: feat
    })
    nn_likelihood = model.likelihood_link.eval(feed_dict={
        model.u: u, model.v: v,
        model.q_pos_fixed: np.full_like(u, alpha, dtype=np.float32),
        model.q_neg_fixed: np.full_like(u, 1 - alpha, dtype=np.float32)
    })
    
    nn_results = {f"link_{metric.__name__}": metric(is_nn_link, nn_likelihood) for metric in METRICS}
    nf_results = {f"feat_{metric.__name__}": metric(is_nf_link, nf_likelihood) for metric in METRICS}
    
    if baselines:
        for baseline_name, baseline in baselines.items():
            scores = [baseline.score(n, f) for n, f in zip(node, feat)]
            for metric in METRICS:
                nf_results[f"{baseline_name}_feat_{metric.__name__}"] = metric(is_nf_link, scores)
    return {
        **nn_results, **nf_results, 
        'log_loss': nn_results['link_log_loss'] + nf_results['feat_log_loss']
    }

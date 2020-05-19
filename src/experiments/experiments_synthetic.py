import plots

import mlflow
import numpy as np
import sklearn.metrics
from tqdm import tqdm

import sys
import traceback

sys.path.append("..")
import model

def generate(N, Q, T, interactions_per_timestep=10,
             actions_per_timestep_per_node=10,
             mu_pos=0.1, mu_neg=0.1,
             consensus_threshold=0.6, backfire_threshold=1.2,
             feature_sharpness=8, fixed_sigma=0.5, seed=None):
    if seed is not None:
        np.random.seed(seed)
    w = np.random.uniform(size=Q) * 2 - 1
    sigma = np.full(Q, fixed_sigma)
    sigmoid = lambda x: 1. / (1 + np.exp(-feature_sharpness * x))
    K = lambda xvt, a: sigmoid(sigma[a] - np.abs(xvt - w[a]))

    x0 = np.random.uniform(size=N) * 2 - 1

    u_v_t_w = []
    v_a_t_w = []

    t = 0
    X = [x0]

    for t in range(T):
        xt = X[-1]

        for v in range(N):
            for _ in range(actions_per_timestep_per_node):
                k = np.array([K(xt[v], a) for a in range(Q)])
                k /= np.sum(k)
                a = np.random.choice(np.arange(Q), p=k)
                v_a_t_w.append( (v, a, t, 1) )

        xtp1 = xt.copy()
        for _ in range(interactions_per_timestep):
            i = np.random.randint(N)
            while True:
                j = np.random.randint(N)
                if i != j: break

            dist = np.abs(xt[i] - xt[j])
            if dist < consensus_threshold:
                xtp1[j] += mu_pos * (xt[i] - xt[j])
                u_v_t_w.append( (i, j, t, 1) )
            if dist > backfire_threshold:
                xtp1[j] += mu_neg * (xt[j] - xt[i])
                u_v_t_w.append( (i, j, t, 1) )
            xtp1 = np.clip(xtp1, -1, 1)

        X.append(xtp1)

    X = np.vstack(X)

    return u_v_t_w, v_a_t_w, X, w

def compare(X_original, X_estimated, w_original, w_estimated):
    mean_diff_t0 =           np.mean(np.abs(X_original[0] -   X_estimated[0] ))
    mean_diff_t0_symmetric = np.mean(np.abs(X_original[0] - (-X_estimated[0])))
    
    if mean_diff_t0_symmetric < mean_diff_t0:
        X_estimated, w_estimated = -X_estimated, -w_estimated
        mean_diff_t0 = mean_diff_t0_symmetric
    
    avg_diffs = [np.mean(np.abs(X_original[t] - X_estimated[t]))
                 for t in range(X_estimated.shape[0])]
    w_diff = np.abs(w_original - w_estimated)
    
    evals = {
        'mean_x0_diff': mean_diff_t0,
        'std_dev_x0_diff': np.std(np.abs(X_original[0] - X_estimated[0])),
        'mean_w_diff': np.mean(w_diff),
        'std_dev_w_diff': np.std(w_diff)
    }
    return evals, avg_diffs

def compare_signs(u_v_t_w, real_X, t2signs_est, backfire_threshold, consensus_threshold, **_ignored_kw):
    dist2sign = lambda x: 1 if x < consensus_threshold else (-1 if x > backfire_threshold else np.nan)
    
    tru_edge2sign = {(u, v, t): dist2sign(np.abs(real_X[t, u] - real_X[t, v])) for u, v, t, w in u_v_t_w}
    est_edge2sign = {(u, v, t) : s for t, signs in enumerate(t2signs_est) for ((u, v), s) in signs}

    tru_sign_sequence = np.array([tru_edge2sign[(u, v, t)] for u, v, t, w in u_v_t_w])
    est_sign_sequence = np.array([est_edge2sign[(u, v, t)] for u, v, t, w in u_v_t_w])
    
    if np.any(np.isnan(tru_sign_sequence)):
        raise Exception("Arcs in the real data set should only contain positive or negative interactions")
    
    return {
        'signs_f1': sklearn.metrics.f1_score(tru_sign_sequence, est_sign_sequence),
        'signs_prec': sklearn.metrics.precision_score(tru_sign_sequence, est_sign_sequence),
        'signs_rec': sklearn.metrics.recall_score(tru_sign_sequence, est_sign_sequence),
    }


hyperparams_settings = {
    'consensus': {'mu_pos': 0.1, 'mu_neg': 0.1, 'consensus_threshold': 1.2, 'backfire_threshold': 1.6},
    'egoinvolv': {'mu_pos': 0.1, 'mu_neg': 0.1, 'consensus_threshold': 0.4, 'backfire_threshold': 0.6},
    'noncommit': {'mu_pos': 0.1, 'mu_neg': 0.1, 'consensus_threshold': 0.2, 'backfire_threshold': 1.6},
    'balkanize': {'mu_pos': 0.1, 'mu_neg': 0.1, 'consensus_threshold': 0.6, 'backfire_threshold': 1.2},
}

hyperparams_names = ['mu_pos', 'mu_neg', 'consensus_threshold', 'backfire_threshold']
assert all(set(h.keys()) == set(hyperparams_names) for h in hyperparams_settings.values())

def main(
    N=30,
    Q=20,
    T=10,
    actions_per_timestep_per_node=15,
    interactions_per_timestep=90,
    num_generations=8,
    num_experiment_per_generation=3,
    fixed_sigma=0.1,
    num_epochs=2,
    alpha_clip_0=0.4,
    threshold=5E-4
):
    for experiment_run_id in range(num_generations):
        for generative_id, generative_setting in tqdm(hyperparams_settings.items()):
            generative_seed = hash(f"{experiment_run_id}-{generative_id}") & (2**32 - 1)
            u_v_t_w, v_a_t_w, X_original, w_original = generate(N=N, Q=Q, T=T,
                                                actions_per_timestep_per_node=actions_per_timestep_per_node,
                                                fixed_sigma=fixed_sigma,
                                                interactions_per_timestep=interactions_per_timestep,
                                                seed=generative_seed,
                                                **generative_setting)
            
            for _ in range(num_experiment_per_generation):
                for estimation_id, estimation_setting in hyperparams_settings.items():
                    print('===========================================================================')
                    print('generative_setting:', generative_id, '\testimation_setting:', estimation_id, '\texperiment_run_id:', experiment_run_id)
                    print('===========================================================================')
                    
                    mlflow.start_run()
                    mlflow.log_param("generative_seed", generative_seed)
                    mlflow.log_param("generative_setting", generative_id)
                    mlflow.log_param("estimation_setting", estimation_id)
                    mlflow.log_param("true_hyperparameters", generative_id == estimation_id)
                    mlflow.log_param("generation_id", f"{generative_id}_{experiment_run_id}")
                    mlflow.log_param("N", N)
                    mlflow.log_param("Q", Q)
                    mlflow.log_param("T", T)
                    mlflow.log_param("interactions_per_timestep", interactions_per_timestep)
                    mlflow.log_param("actions_per_timestep_per_node", actions_per_timestep_per_node)
                    mlflow.log_param("num_epochs", num_epochs)
                    mlflow.log_param("alpha_clip_0", alpha_clip_0)
                    for k, v in generative_setting.items():
                        mlflow.log_param('generation_' + k, v)
                    for k, v in estimation_setting.items():
                        mlflow.log_param('estimation_' + k, v)
                    
                    try:
                        
                        X_estimated, w_estimated, _sigma, t2signs_estimated, alphas, evals = model.learn_opinion_dynamics(
                                    N=N, Q=Q, T=T, verbose=False,
                                    u_v_t_weights=u_v_t_w, v_a_t_weights=v_a_t_w,
                                    num_epochs=num_epochs, threshold=threshold, alpha_clip_0=alpha_clip_0,
                                    **estimation_setting)
                                                        
                        comparison, avg_diffs = compare(X_original, X_estimated, w_original, w_estimated)
                        comparison.update(compare_signs(u_v_t_w, X_original, t2signs_estimated, **generative_setting))
                        
                        for key, val in list(evals.items()) + list(comparison.items()):
                            mlflow.log_metric(key, val)
                        plots.make_plots(X_original, X_estimated, w_original, w_estimated)
                        
                        np.savetxt('estimated_alphas.txt', alphas)
                        mlflow.log_artifact('estimated_alphas.txt')
                        np.savetxt('avg_diffs.txt', avg_diffs)
                        mlflow.log_artifact('avg_diffs.txt')
                        
                    except Exception as e: # pylint: disable=broad-except
                        mlflow.set_tag('crashed', True)
                        mlflow.set_tag('exception', e)
                        with open("exception.txt", 'w') as f:
                            traceback.print_exc(file=f)
                        mlflow.log_artifact("exception.txt")
                        
                    mlflow.end_run()

if __name__ == '__main__':
    mlflow.set_experiment("LODM_synthetic")
    main()

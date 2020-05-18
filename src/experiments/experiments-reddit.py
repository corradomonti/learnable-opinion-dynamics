import exoevals
import reddit

import mlflow
import numpy as np
import pandas as pd

import pickle
import sys
import traceback

sys.path.append("..")
import model


hyperparams_settings = {
    'consensus': {'consensus_threshold': 1.2, 'backfire_threshold': 1.6},
    'egoinvolv': {'consensus_threshold': 0.4, 'backfire_threshold': 0.6},
    'noncommit': {'consensus_threshold': 0.2, 'backfire_threshold': 1.6},
    'balkanize': {'consensus_threshold': 0.6, 'backfire_threshold': 1.2},
}

T = 120

user2id, subreddit2id, u_v_t_weights, v_a_t_weights, u_v_test, v_a_test = reddit.read_reddit(T)

def main(
    params,
    fixed_subreddits,
    num_experiment_per_setting=4,
):
    for experiment_run_id in range(num_experiment_per_setting):
        for fixed_subreddit in fixed_subreddits:
            if fixed_subreddit is not None:
                fixed_features = {subreddit2id[sub.encode()]: w for sub, w in fixed_subreddit.items()}
            else:
                fixed_features = None

            for setting_id, setting in hyperparams_settings.items():
                print('===========================================================================')
                print('REDDIT\tsetting:', setting_id, '\texperiment_run_id:', experiment_run_id)
                print('===========================================================================')

                mlflow.start_run()
                mlflow.log_param("experiment_run_id", experiment_run_id)
                mlflow.log_param("fixed_subreddit", fixed_subreddit)
                mlflow.log_param("setting", setting_id)

                params_and_setting = {**params, **setting}
                for k, v in params_and_setting.items():
                    mlflow.log_param(k, v)

                try:
                    X, w, sigma, t2signs, alphas, evals = model.learn_opinion_dynamics(
                                N=len(user2id), Q=len(subreddit2id), T=T,
                                u_v_t_weights=u_v_t_weights, v_a_t_weights=v_a_t_weights,
                                verbose=False, fixed_features=fixed_features,
                                **params_and_setting)

                    with open("results.pickle", 'wb') as f:
                        pickle.dump((X, w, sigma, t2signs, alphas), f)
                    mlflow.log_artifact('results.pickle')

                    exogen_evals = exoevals.make_exogenous_evals(X, w, sigma, user2id, subreddit2id)

                    for key, val in list(evals.items()) + list(exogen_evals.items()):
                        mlflow.log_metric(key, val)

                    np.savetxt('estimated_alphas.txt', alphas)
                    mlflow.log_artifact('estimated_alphas.txt')

                    pd.DataFrame([[user.decode(), i, X[0][i]] for (user, i) in user2id.items()],
                                  columns=["user", "id", "X0"]).sort_values(by='X0').to_csv(
                                  "user_opinions.csv", index=False)
                    mlflow.log_artifact('user_opinions.csv')

                    pd.DataFrame([[sub.decode(), i, w[i], sigma[i]] for (sub, i) in subreddit2id.items()],
                                  columns=["subreddit", "id", "w", "sigma"]).sort_values(by='w').to_csv(
                                  "subreddit_opinions.csv", index=False)
                    mlflow.log_artifact('subreddit_opinions.csv')

                except Exception as e: # pylint: disable=broad-except
                    mlflow.set_tag('crashed', True)
                    mlflow.set_tag('exception', e)
                    with open("exception.txt", 'w') as f:
                        traceback.print_exc(file=f)
                    mlflow.log_artifact("exception.txt")

                mlflow.end_run()

if __name__ == '__main__':
    mlflow.set_experiment("LODM_reddit")
    main(params={
            'num_epochs': 2,
            'threshold': 5E-4,
            'alpha_clip_0': 0.4,
            'link_sharpness': 16,
            'feature_sharpness': 8,
            'mu_pos': 0.001,
            'mu_neg': 0.001,
            'sigma_prior_p': 8.,
            'sigma_prior_q': 8.,
            'learning_rate_link': 0.0001,
            'learning_rate_feature': 0.001,
            'sigma_prior_coefficient': 1.
        }, fixed_subreddits=[
           None,
           {'democrats': -1, 'Republican': 1},
           {'The_Donald': 1},
       ])

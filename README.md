# Learning Opinion Dynamics From Social Traces

This repository contains code and data related to the paper ***"Learning Opinion Dynamics From Social Traces"*** by Corrado Monti, Gianmarco De Francisci Morales and Francesco Bonchi, published at KDD 2020. If you use the provided data or code, we would appreciate a citation to the paper:

```
@inproceedings{monti2020learningopinion,
  title={Learning Opinion Dynamics From Social Traces},
  author={Monti, Corrado and De Francisci Morales, Gianmarco and Bonchi, Francesco},
  booktitle={Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
  year={2020}
}
```

Here you will find instructions about (i) using the algorithm we presented and implemented (ii) obtaining the Reddit data set we collected (iii) how to reproduce our experiments.

## Model implementation

In order to use our model, we provide our implementation in `src/model`. To install it, clone this repo, install the dependencies, and install our model:

```
git clone https://github.com/corradomonti/learnable-opinion-dynamics.git
cd learnable-opinion-dynamics
pip install -r requirements.txt
pip install src
```

Then you can import it into Python with:

```
>>> from model import learn_opinion_dynamics
```

Check its documentation with:

```
>>> help(learn_opinion_dynamics)
```

For instance, to use it on a simple graph, with 3 nodes, 2 features and 2 time steps, and then print the estimated positions of each feature:

```
>>> u_v_t_weights = ([0, 1, 0, 1], [1, 0, 1, 1])
>>> v_a_t_weights = ([0, 0, 0, 1], [2, 1, 0, 1], [0, 0, 1, 1], [2, 1, 1, 1])
>>> res = learn_opinion_dynamics(N=3, Q=2, T=2, u_v_t_weights=u_v_t_weights, v_a_t_weights=v_a_t_weights)
>>> print(res.w)
```

## Provided data set

`data/reddit` contains the Reddit data set we gathered. 
All the data files are TSV compressed with `bz2` and can be easily opened with pandas.
Please note that the input files (`edges_user.tsv.bz2`, `edges_feature.tsv.bz2`) do not have a header. To parse them you can also use the code we provide in `src/experiments/reddit.py`.

To build this data, we consider the 51 subreddits most similar to `r/politics` according to [this](https://www.shorttails.io/interactive-map-of-reddit-and-subreddit-similarity-calculator); the time stamps are the months between January 2008 and December 2017; for the users, we consider only those posting a minimum of 10 comments per month on r/politics for at least half of the considered months, which gives us 375 users.

Our input is:

- `edges_user.tsv.bz2` contains the interactions among considered users. Each row (t, u, v, w) indicates that user v replied to user u during time step t, for w times.

- `edges_feature.tsv.bz2` contains the user-subreddit interactions. Each row (t, u, a, w) indicates that user u participated in subreddit a during time step t, for w times.

Our validation data is:

- `feature_scores.tsv.bz2` contains the summary statistics for scores received by each user on each subreddit in each timestep. Specifically, it contains the sum of positive scores, the number of positively scored comments, the sum of negative scores, and the number of negatively scored comments.

- `interaction_scores.tsv.bz2` contains data about each interaction between considered users.

## Reproducibility

In order to reproduce our experiments, we provide our scripts in `src/experiments`. They need a larger set of dependencies, listed in `src/experiments/requirements.txt`. In particular, we use [MLflow](https://github.com/mlflow/mlflow) to organize parameters and experiment results. To run both sets of experiments, do:

```
cd src/experiments/
pip install -r requirements.txt
python experiments-synthetic.py
python experiments-reddit.py
mlflow ui
```

Then, you can use the MLflow User Interface to inspect the results of each experiment.

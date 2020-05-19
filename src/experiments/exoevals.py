from reddit import DATASET_PATH, timestamp2t
import plots

import pandas as pd
import numpy as np

import scipy.stats

def make_exogenous_evals(X, w, sigma, user2id, subreddit2id):
    exogen_evals = dict()
    def add_with_pvalue(key, value, pvalue):
        exogen_evals[key] = value
        exogen_evals[key + '-pvalue'] = pvalue
        print(key, value, pvalue)

    feature_scores = pd.read_csv(DATASET_PATH + 'feature_scores.tsv.bz2', sep='\t').astype({'date': str})
    feature_scores['t'] = list(map(timestamp2t, feature_scores.date))
    feature_scores['t_year'] = feature_scores.t // 12
    feature_scores = feature_scores.groupby(['author', 'subreddit', 't_year'], as_index=False).sum()

    feature_scores['avg_score'] = (feature_scores.sum_p + feature_scores.sum_n) / (
        feature_scores.count_p + feature_scores.count_n)

    def find_our_feature_score(author, subreddit, t):
        try:
            return abs(X[t, user2id[author.encode()]] - w[subreddit2id[subreddit.encode()]])
        except (KeyError, IndexError):
            return np.nan

    feature_scores['our_distance'] = [
        find_our_feature_score(v, a, t) for v, a, t in zip(
            feature_scores.author, feature_scores.subreddit, feature_scores.t_year
        )
    ]

    min_score = feature_scores.avg_score.min()
    normalization = lambda x: np.log1p(x - min_score)
    feature_scores['avg_score_normalized'] = normalization(feature_scores.avg_score)


    only_within_two_sigma = (
        np.abs(feature_scores.avg_score_normalized - np.mean(feature_scores.avg_score_normalized)) / 
        np.std(feature_scores.avg_score_normalized)
    ) < 2

    add_with_pvalue('user-subreddit-pearsonr', *scipy.stats.pearsonr(
                        feature_scores.avg_score_normalized,
                         feature_scores.our_distance))

    add_with_pvalue('user-subreddit-two-sigma-pearsonr', *scipy.stats.pearsonr(
        feature_scores.avg_score_normalized[only_within_two_sigma],
        feature_scores.our_distance[only_within_two_sigma]))

    add_with_pvalue('user-subreddit-two-sigma-spearmanr', *scipy.stats.spearmanr(
        feature_scores.avg_score, feature_scores.our_distance))
    
    plots.reddit_regression(feature_scores, normalization, ["reddit_distance.pdf", "reddit_distance.png"])

    
    interaction_scores = pd.read_csv(DATASET_PATH + "interaction_scores.tsv.bz2", sep='\t')

    interaction_scores['min_score'] = np.minimum(np.abs(interaction_scores.score_x),
                                                       np.abs(interaction_scores.score_y))

    interaction_scores['is_fighting'] = (interaction_scores.score_x * interaction_scores.score_y) < 0
    interaction_scores['non_fighting'] = (interaction_scores.score_x > 0) & (interaction_scores.score_y > 0)

    def find_node_node_distance(username_u, username_v, timestamp):
        try:
            t = timestamp2t(str(timestamp))
            return abs(X[t, user2id[username_u.encode()]] - X[t, user2id[username_v.encode()]])
        except (KeyError, IndexError):
            return np.nan

    interaction_scores['our_distance'] = [
            find_node_node_distance(u, v, t)
            for u, v, t in zip(
                 interaction_scores.author_x, interaction_scores.author_y, interaction_scores.date)
    ]

    threshold = 10

    fighting = interaction_scores.is_fighting & (interaction_scores.min_score > threshold)
    non_fighting = interaction_scores.non_fighting & (interaction_scores.min_score > threshold)
    
    add_with_pvalue('distance-fighting', *scipy.stats.mannwhitneyu(
        interaction_scores.our_distance[fighting],
        interaction_scores.our_distance[non_fighting],
        alternative='greater'
    ))
    plots.reddit_conflict(interaction_scores.our_distance, fighting, non_fighting, ["reddit_conflict.pdf", "reddit_conflict.png"])

    fighting_median = np.median(interaction_scores.our_distance[fighting])
    non_fighting_median = np.median(interaction_scores.our_distance[non_fighting])
    exogen_evals['distance-fighting-median'] = fighting_median
    exogen_evals['distance-non-fighting-median'] = non_fighting_median
    return exogen_evals

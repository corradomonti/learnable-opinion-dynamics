import matplotlib.pyplot as plt
import mlflow
import numpy as np
import seaborn as sns

def scatterplot_estimate(x, x_est, output_path, variable_name):
    plt.figure(figsize=(5, 5))
    sns.scatterplot(x, x_est)#, bins=np.arange(-1, 1, 0.1), cmap='afmhot')
    plt.xlim((-1.05, 1.05))
    plt.ylim((-1.05, 1.05))
    plt.xlabel(r"Original $\mathbf{%s}$" % variable_name)
    plt.ylabel(r"Estimated $\mathbf{%s}$" % variable_name)
    plt.savefig(output_path)
    plt.close()
    mlflow.log_artifact(output_path)

def plot_opinions_in_time(X, output_path):
    plt.figure(figsize=(10, 5))
    plt.plot(X, '-')#, bins=np.arange(-1, 1, 0.1), cmap='afmhot')
    plt.ylim((-1.05, 1.05))
    plt.xlabel("time")
    plt.ylabel(r"$\mathbf{x}$")
    plt.savefig(output_path)
    plt.close()
    mlflow.log_artifact(output_path)

def make_plots(X, X_est, w, w_est, path=''):
    sns.set()
    sns.set_style("whitegrid")
    plot_opinions_in_time(X, path + "original.png")
    plot_opinions_in_time(X_est, path + "estimate.png")
    scatterplot_estimate(X[0], X_est[0], path + "scatter.png", 'x_0')
    scatterplot_estimate(w, w_est, path + "scatter-w.png", 'w')

def reddit_conflict(our_distance, fighting, non_fighting, output_paths):
    sns.set()
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.45)
    # plt.figure(figsize=(8, 8))

    sns.distplot(our_distance[non_fighting],
                 bins=np.arange(0, 0.5, 0.025),
                 norm_hist=True, kde=False, label="Non-conflictual interactions")
    sns.distplot(our_distance[fighting],
                 bins=np.arange(0, 0.5, 0.025),
                 norm_hist=True, kde=False, label="Conflictual interactions")
    plt.legend()
    frame1 = plt.gca()
    frame1.axes.yaxis.set_ticklabels([])
    plt.xlabel("Model-predicted distance")
    plt.ylabel("PDF")
    for output_path in output_paths:
        plt.savefig(output_path, bbox_inches='tight')
        mlflow.log_artifact(output_path)
    plt.close()

def reddit_regression(feature_scores, normalization, output_paths):
    unnormalized_ticks = np.arange(2, 8, 0.5)
    sns.regplot(y="avg_score_normalized", x="our_distance",
            data=feature_scores,
            x_bins=10,
            x_ci='ci')
    plt.yticks(normalization(unnormalized_ticks), unnormalized_ticks)
    plt.ylabel("User-subreddit score")
    plt.xlabel("Model-predicted distance")
    for output_path in output_paths:
        plt.savefig(output_path, bbox_inches='tight')
        mlflow.log_artifact(output_path)
    plt.close()

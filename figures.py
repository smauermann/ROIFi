import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_figure(labels_values, parameter_labels, method, age_corr,
                normalized, mean_cluster_sizes=None,
                all_n_clusters=None, regression=None):
    sns.set_context("notebook", font_scale=1.2)
    corr_str = 'age-corrected' if age_corr else 'not age-corrected'
    atlas_score = 0.70211 if age_corr else 0.68570
    norm_str = 'normalized' if normalized else 'unnormalized'
    x_label_str = 'n clusters' if method == 'SpectralClustering' else 'r threshold'

    atlas_score = 0.70211 if age_corr else 0.68570
    labels_values = np.array(labels_values)
    if labels_values.ndim == 2:
        parameter_labels = labels_values[:, 0]
        class_scores = labels_values[:, 1]
    else:
        class_scores = labels_values
    # things to plot
    # classification scores
    things_to_plot = [sorted(list(zip(parameter_labels, class_scores)))]
    # mean_cluster_sizes
    if mean_cluster_sizes is not None:
        things_to_plot.append(sorted(list(zip(parameter_labels,
                                              mean_cluster_sizes))))
    # all_n_clusters
    if all_n_clusters is not None:
        things_to_plot.append(sorted(list(zip(parameter_labels,
                                              all_n_clusters))))

    n_subplots = len(things_to_plot)
    fig, axes = plt.subplots(n_subplots, sharex=True)
    st = fig.suptitle('%s: %s, %s' % (method, corr_str, norm_str))

    for i, t in enumerate(things_to_plot):
        data = np.array(t).T
        x, y = data
        if n_subplots > 1:
            axes[i].plot(x, y, color='g', markersize=5)
        else:
            axes.axhline(y=atlas_score, color='red', linestyle='--', linewidth=0.7)
            axes.plot(x, y, color='g')
    if n_subplots == 3:
        axes[2].set_xlim(0.1, np.max(x))
        axes[0].set_ylabel('balanced\naccuracy')
        axes[0].set_ylim(0.53, 0.73)
        axes[1].set_ylabel('mean cluster\nsize')
        axes[1].set_ylim(-500, 12000)
        axes[2].set_ylabel('n clusters')
        axes[2].set_ylim(0, 35)
        axes[2].set_xlabel(x_label_str)
    else:
        #axes.set_ylim(0.66, 0.725)
        axes.set_xlim(0.75, np.max(x))
        axes.set_ylabel('balanced accuracy')
        axes.set_xlabel(x_label_str)

    if regression is not None:
        sns.regplot(x=x, y=y, scatter=False, order=regression,
                    line_kws={"color": "red"})

    return fig

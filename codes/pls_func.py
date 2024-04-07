"""
*******************************************************************************

Function for plotting the results of behavioral PLS           

*******************************************************************************
"""

#------------------------------------------------------------------------------
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from scipy.stats import pearsonr
import matplotlib.patches as mpatches
#------------------------------------------------------------------------------

def plot_scores_and_correlations_unicolor(lv,
                                          pls_result,
                                          title,
                                          clinical_scores,
                                          path_fig,
                                          column_name):

    plt.figure(figsize = (8, 8))
    plt.scatter(range(1, len(pls_result.varexp) + 1),
                pls_result.varexp)
    plt.savefig(path_fig +'VarianceExplanined.svg',
            bbox_inches = 'tight', dpi = 300,
            transparent = True)
    plt.title(title)
    plt.xlabel('Latent variables')
    plt.ylabel('Variance Explained')

    # Calculate and print singular values
    singvals = pls_result["singvals"] ** 2 / np.sum(pls_result["singvals"] ** 2)
    print(f'Singular values for latent variable {lv}: {singvals[lv]:.4f}')

    # Plot score correlation
    plt.figure(figsize = (8, 8))
    plt.title(f'Scores for latent variable {lv}')

    # Assuming 'Diagnosis' column holds values "ALS" or "control"
    sns.regplot(x = pls_result['x_scores'][:, lv],
                y = pls_result['y_scores'][:, lv],
                scatter = False)
    sns.scatterplot(x = pls_result['x_scores'][:, lv],
                    y = pls_result['y_scores'][:, lv],
                    c = clinical_scores,
                    s=200,
                    cmap = 'coolwarm',
                    vmin = 0,
                    vmax = 1,
                    edgecolor='black',
                    linewidth = 0.5)

    plt.xlabel('X scores')
    plt.ylabel('Y scores')
    plt.tight_layout()

    plt.savefig(path_fig +'score_xy_' + str(lv) + '_' + column_name + '.svg',
            bbox_inches = 'tight',
            dpi = 300,
            transparent = True)

    # Calculate and print score correlations
    score_correlation_spearmanr = spearmanr(pls_result['x_scores'][:, lv],
                                            pls_result['y_scores'][:, lv])
    score_correlation_pearsonr = pearsonr(pls_result['x_scores'][:, lv],
                                          pls_result['y_scores'][:, lv])

    print(f'x-score and y-score Spearman correlation for latent variable {lv}: \
          {score_correlation_spearmanr.correlation:.4f}')
    print(f'x-score and y-score Pearson correlation for latent variable {lv}: \
          {score_correlation_pearsonr[0]:.4f}')

#------------------------------------------------------------------------------ 
def plot_loading_bar(lv,
                     pls_result,
                     combined_columns,
                     column_groups,
                     group_colors,
                     vmin_val,
                     vmax_val,
                     path_fig):
    err = (pls_result["bootres"]["y_loadings_ci"][:, lv, 1] -
           pls_result["bootres"]["y_loadings_ci"][:, lv, 0]) / 2
    values = pls_result.y_loadings[:,lv]

    # Create a bar plot with different colors
    plt.figure(figsize = (10,16)) # Adjust the figure size if needed
    bars = plt.barh(combined_columns,
                   values,
                   xerr = err)

    # Adding labels and title
    plt.xlabel('x-loading')
    plt.ylabel('behavioral measure')

    # Set the color of each bar based on the group
    for bar, column_name in zip(bars, combined_columns):
        group = [group 
                 for group, cols in column_groups.items()
                 if column_name in cols][0]
        bar.set_color(group_colors.get(group, 'gray'))

    # Create custom legend handles and labels
    legend_handles = [mpatches.Patch(color = group_colors[group],
                                     label = group) for group in group_colors]
    plt.legend(handles = legend_handles)
    # Remove the box around the figure
    for spine in plt.gca().spines.values():
        spine.set_visible(False)

    ax = plt.gca()

    # Generate x-axis ticks including the maximum value
    x_ticks = np.linspace(vmin_val, vmax_val, num = 5)  # 5 ticks from 0 to x_max

    ax.set_xticks(x_ticks)

    plt.xticks(rotation = 90)
    plt.tight_layout()

    plt.savefig(path_fig + 'barplot_' + str(lv) +'.svg',
            bbox_inches = 'tight',
            dpi = 300,
            transparent = True)

    plt.show()
#------------------------------------------------------------------------------
# END
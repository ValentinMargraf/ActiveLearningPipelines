import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize, LinearSegmentedColormap

import pandas as pd
from py_experimenter.experimenter import PyExperimenter
import json
import os
from scipy.stats import ttest_ind_from_stats
from scipy.stats import ttest_ind as tt


qs_for_figure = {"margin": "MS", "least_confident": "LC", "entropy": "ES",
                 "power_margin": "PowMS", "bald": "BALD", "power_bald": "PowBALD",
                 "max_entropy": "MaxEnt", "qbc_variance_ratio": "QBC VR",
                 "core_set": "CoreSet", "typ_cluster": "TypClu", "cluster_margin": "CluMS",
                 "weighted_cluster": "Clue", "falcun": "FALCUN", "random": "Rand",
                 "epistemic": "EU", "aleatoric": "AU", "switching": "switch", "total": "total"}
info = ["entropy", "least_confident", "margin", "power_margin",
        "max_entropy", "bald", "power_bald", "qbc_variance_ratio", "epistemic",
        "aleatoric", "total", "switching"]
repr = ["kmeans", "core_set", "typ_cluster"]
hybr = ["cluster_margin", "falcun", "weighted_cluster"]
stacked_qs = [info, repr, hybr]
all_qs_ordered = [x for xs in stacked_qs for x in xs]
all_qs_ordered.insert(0, "random")

learner_for_figure = {"knn_3": "KNN", "svm_rbf": "SVM (rbf)", "mlp": "MLP", "rf_entropy": "RF (entr)",
                      "rf_gini": "RF (gini)", "svm_lin": "SVM (lin)", "catboost": "Catboost", "xgb": "XGB",
                      "tabnet": "Tabnet", "tabpfn": "TabPFN"}
all_learners_ordered = ["knn_3", "svm_rbf", "rf_entropy", "catboost",
                    "xgb", "mlp",
                    "tabnet",
                    "tabpfn"]

class BudgetPerformancePlot:
    """BudgetPerformancePlot

    This class plots the performance for a given learning algorithm and openmlid of different query strategies over
    the whole active learning procedure resulting in so-calles area under the budget curves (AUBC).

    Args:
        df (pd.DataFrame): The dataframe.
        openml_id (int): The openml id.
        learner_name (str): The learner name.
        metric (str): The metric.
        path_to_save (str): The path to save the plot.

    Attributes:
        df (pd.DataFrame): The dataframe.
        openml_id (int): The openml id.
        learner_name (str): The learner name.
        metric (str): The metric.
        path_to_save (str): The path to save the plot.
        plot_data (dict): The data to plot.
        num_seeds (int): The number of seeds (used to determine the std error).


    """

    def __init__(self, df, openml_id, learner_name, metric, path_to_save=None):
        self.df = df
        self.openml_id = openml_id
        self.learner_name = learner_name
        self.metric = metric
        self.path_to_save = path_to_save
        self.plot_data = None
        self.num_seeds = None

    def generate_plot_data(self):
        """
        This function generates the data to plot.
        """
        # get data for openml_id
        df = self.df[self.df['openml_id'] == self.openml_id]
        # get data for learner
        df = df[df['learner_name'] == self.learner_name]
        # get unique query strategy names
        query_strategies = df['query_strategy_name'].unique()
        # get unique budget values
        budgets = df['len_X_l'].unique()
        # create dict to store data
        data = {}
        for qs in query_strategies:
            data[qs] = {'budget': [], 'mean': [], 'std': []}
            for enum, budget in enumerate(budgets):
                # get data for query strategy and budget
                df_temp = df[(df['query_strategy_name'] == qs) & (df['len_X_l'] == budget)]

                # reset index
                df_temp.reset_index()
                # get mean and std of metric
                mean = df_temp[self.metric].mean()
                std = df_temp[self.metric].std()

                df_reset = df_temp[self.metric].reset_index()
                # append data to dict
                data[qs]['budget'].append(budget)
                data[qs]['mean'].append(mean)
                data[qs]['std'].append(std)

        # get num of seeds
        self.num_seeds = len(df['seed'].unique())
        self.plot_data = data

    def show(self, show_fig=False):
        """
        This function plots the performance of different query strategies over the budget and saves
        it as .pdf under the specified path.
        """
        data = self.plot_data
        if len(data.keys()) == 0:
            return
        else:
            # we want to order the QS by grouping them into categories, also each category gets a different
            # color coding (uncertainty based are redish, representative are greenish, hybrid are blueish,
            # random is pink)
            keys = data.keys()
            list_of_qs = list(keys)
            # order the query strategies
            ordered_qs = []
            for qs in all_qs_ordered:
                if qs in list_of_qs:
                    ordered_qs.append(qs)

            fig, ax = plt.subplots(1)
            # colormap of len(keys)
            colors = plt.cm.tab20(np.linspace(0, 1, len(keys)))
            color_dict = {"random": "magenta", "least_confident": "rosybrown", "margin": "red",
                          "entropy": "orange", "power_margin": "brown", "bald": "tomato",
                          "power_bald": "coral", "max_entropy": "sandybrown", "qbc_variance_ratio": "peachpuff",
                          "core_set": "limegreen", "typ_cluster": "forestgreen", "cluster_margin": "mediumblue",
                          "weighted_cluster": "turquoise", "falcun": "blue", "epistemic": "black", "aleatoric": "gray",
                          "switching": "green", "total": "yellow"}
            for cl, key in enumerate(ordered_qs):
                budget = np.array(data[key]['budget'])
                mu = np.array(data[key]['mean'])
                # std error
                std = np.array(data[key]['std']) / np.sqrt(self.num_seeds)
                qs = key
                cl = color_dict[qs]
                plt.plot(budget, mu, lw=2, label=key, color=cl)
                plt.fill_between(budget, mu + std, mu - std, facecolor=cl, alpha=0.5)



            handles, labels = plt.gca().get_legend_handles_labels()

            new_labels = []
            for label in labels:
                new_labels.append(qs_for_figure[label])

            plt.title(learner_for_figure[self.learner_name] + " on id " + str(self.openml_id), fontsize=30)
            plt.legend(handles, new_labels, fontsize=20,
                       loc='lower right')  # fontsize=25,loc='center left', bbox_to_anchor=(1, 0.5))
            plt.yticks(fontsize=20)
            plt.xticks(fontsize=20)
            ax.set_xlabel('Number of labeled instances', fontsize=25)
            ax.set_ylabel('test accuracy', fontsize=25)
            [l.set_visible(False) for (i, l) in enumerate(ax.xaxis.get_ticklabels()) if i % 2 != 0]

            # save image
            if self.path_to_save is not None:
                if not os.path.exists(self.path_to_save):
                    os.makedirs(self.path_to_save)
            else:
                self.path_to_save = "FIGURES/BUDGET_PERFORMANCE_PLOT/" + str(self.openml_id) + "/"
                if not os.path.exists(self.path_to_save):
                    os.makedirs(self.path_to_save)
            self.path_to_save = self.path_to_save + str(self.learner_name) + ".pdf"
            fig.savefig(self.path_to_save, facecolor='white', transparent=True, bbox_inches='tight')
            if show_fig:
                plt.show()
            plt.close()





class WinMatrixPlot:
    def __init__(self, df, learner_name, path_to_save = None, statistical_significant=True):
        self.df = df
        self.learner_name = learner_name
        self.path_to_save = path_to_save
        self.statistical_significant = statistical_significant
        self.setting_name = df["setting_name"].unique()[0]
        self.win_matrix = None
        self.query_strategies = None
        self.num_datasets = None

    def generate_win_matrix(self):
        """
        This function generates win-matrices to compare performances of different query strategies combined
        with one fixed learning algorithm.
        """
        self.win_matrix = {}
        df = self.df[self.df['learner_name'] == self.learner_name]
        self.query_strategies = self.df['query_strategy_name'].unique()
        for qs1 in self.query_strategies:
            for qs2 in self.query_strategies:
                self.win_matrix[(qs1, qs2)] = [0, 0]
        thresh = 0.05 if self.statistical_significant else 1.01
        self.num_datasets = len(df['openml_id'].unique())
        for oid in df['openml_id'].unique():
            oid_df = df[df['openml_id'] == oid]
            for qs1 in self.query_strategies:
                for qs2 in self.query_strategies:
                    if qs1 != qs2:
                        df1 = oid_df[oid_df['query_strategy_name'] == qs1]
                        df2 = oid_df[oid_df['query_strategy_name'] == qs2]
                        if not df1.empty and not df2.empty:

                            mean1 = df1['aubc'].mean()
                            mean2 = df2['aubc'].mean()
                            std1 = df1['aubc'].std()
                            std2 = df2['aubc'].std()

                            t, p = ttest_ind_from_stats(mean1, std1, len(df1), mean2, std2, len(df2))
                            if p < thresh:
                                if mean1 > mean2:
                                    self.win_matrix[(qs1, qs2)][0] += 1
                                else:
                                    self.win_matrix[(qs1, qs2)][1] += 1




    def show(self, show_fig=False):
        """
        This function generates win-matrices to compare performances of different query strategies combined
        with one fixed learning algorithm.
        """

        qs_ordered = []
        for qs in all_qs_ordered:
            if qs in self.query_strategies:
                qs_ordered.append(qs)

        # create numpy array from dict
        res_wins = np.zeros((len(qs_ordered), len(qs_ordered)))
        for i,qs1 in enumerate(qs_ordered):
            for j,qs2 in enumerate(qs_ordered):
                res_wins[i,j] = self.win_matrix[(qs1, qs2)][0]

        greens = plt.cm.Greens
        reds = plt.cm.Reds
        blues = plt.cm.Blues
        purples = plt.cm.Purples

        # Normalize the data
        norm = Normalize(vmin=res_wins.min(), vmax=res_wins.max()/2)

        # Apply the custom colormap to the data, separate for each group
        red_colors = reds(norm(res_wins))
        green_colors = greens(norm(res_wins))
        blue_colors = blues(norm(res_wins))
        purple_colors = purples(norm(res_wins))

        fig, ax = plt.subplots(figsize=(1.5 * len(qs_ordered), 1.5 * len(qs_ordered)))

        # Zeichnen der Tafel mit den Farben basierend auf den Werten
        for i,qs1 in enumerate(qs_ordered):
            for j,qs2 in enumerate(qs_ordered):
                wins = self.win_matrix[(qs1, qs2)][0]
                ax.text(j + 0.5, i + 0.5,
                        str(int(wins)) + "/" + str(int(self.num_datasets)),
                        ha='center', va='center', color='white',
                        fontsize=25,
                        weight='bold')
                if qs1 in info:
                    colors = red_colors
                elif qs1 in repr:
                    colors = green_colors
                elif qs1 in hybr:
                    colors = blue_colors
                else:
                    colors = purple_colors

                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=True, color=colors[i, j]))

        # adjusting plot
        x_pos = np.arange(1, len(qs_ordered) + 1.5, 1) - .5
        y_pos = np.arange(1, len(qs_ordered) + 1.5, 1) - .5
        x_pos[-1] -= .5
        y_pos[-1] -= .5

        ax.set_xticks(x_pos)
        ax.set_yticks(y_pos)

        qs_names = [qs_for_figure[qs] for qs in qs_ordered]
        qs_names.append(" ")

        ax.set_xticklabels(qs_names, fontsize=40, rotation=45)
        ax.set_yticklabels(qs_names, fontsize=40)
        ax.set_title("Setting: " + self.setting_name + ", Learner: " + self.learner_name,
                     fontsize=50)


        if self.path_to_save is None:
            self.path_to_save = "FIGURES/WIN_MATRICES"
        PATH = self.path_to_save + "/"
        if not os.path.exists(PATH):
            os.makedirs(PATH)
        SAVE_PATH = PATH + self.setting_name +  "_AUBC_"+str(self.learner_name) + ".pdf"
        fig.savefig(SAVE_PATH, facecolor='white', transparent=True, bbox_inches='tight')
        if show_fig:
            plt.show()
        plt.close()


class HeatMapPlot:
    """HeatMapPlot

    This class plots a heatmap of the performance of different active learning pipelines as well as win or lose-
    matrices for the specified learner comparing different query strategies.

    Args:
        data (dict): The data to plot.
        path_to_save (str): The path to save the plot.
        filter_ids (str): The filter ids.
        take_statistical_insignificant (bool): Whether to take statistical insignificant values.

    Attributes:
        data (dict): The data to plot.
        path_to_save (str): The path to save the plot.
        filter_ids (str): The filter ids.
        take_statistical_insignificant (bool): Whether to take statistical insignificant values.

    """
    def __init__(self, df, path_to_save = None, statistical_significant=True, filter_ids="all"):
        self.df = df
        self.path_to_save = path_to_save
        self.statistical_significant = statistical_significant
        self.filter_ids = filter_ids
        self.setting_name = df["setting_name"].unique()[0]
        self.heatmap = None
        self.query_strategies = None
        self.learners = None
        self.num_datasets = None


    def generate_heatmap(self):
        """
        This function generates heatmaps to compare performances of different query strategies combined
        with one fixed learning algorithm.
        """
        self.heatmap = {}
        df = self.df
        self.query_strategies = self.df['query_strategy_name'].unique()
        self.learners = self.df['learner_name'].unique()
        for l in self.learners:
            for qs in self.query_strategies:
                self.heatmap[(l, qs)] = 0
        self.num_datasets = len(df['openml_id'].unique())
        for oid in df['openml_id'].unique():
            oid_df = df[df['openml_id'] == oid]
            # Dictionary to store mean performance per pipeline (learner + query strategy)
            pipeline_means = {}
            for l in self.learners:
                for qs in self.query_strategies:
                    pipeline_df = oid_df[(oid_df['learner_name'] == l) & (oid_df['query_strategy_name'] == qs)]
                    if not pipeline_df.empty:
                        pipeline_means[(l, qs)] = pipeline_df['aubc'].mean()

            # Find the pipeline with the highest mean
            best_pipeline = max(pipeline_means, key=pipeline_means.get)
            self.heatmap[best_pipeline] += 1
            # Increment pipelines that are not statistically significantly worse
            if self.statistical_significant:
                best_mean = pipeline_means[best_pipeline]
                for pipeline, mean in pipeline_means.items():
                    if pipeline != best_pipeline:
                        # Retrieve the statistics for the two pipelines
                        df1 = oid_df[(oid_df['learner_name'] == best_pipeline[0]) & (
                                    oid_df['query_strategy_name'] == best_pipeline[1])]
                        df2 = oid_df[(oid_df['learner_name'] == pipeline[0]) & (
                                    oid_df['query_strategy_name'] == pipeline[1])]

                        if not df1.empty and not df2.empty:
                            mean1 = df1['aubc'].mean()
                            mean2 = df2['aubc'].mean()
                            std1 = df1['aubc'].std()
                            std2 = df2['aubc'].std()
                            t, p = ttest_ind_from_stats(mean1, std1, len(df1), mean2, std2, len(df2))
                            if p < 0.05:
                                self.heatmap[pipeline] += 1


    def show(self, show_fig=False):
        """
        This function plots heatmaps to compare performances of different active learning pipelines.
        The figures are saved under the specified path.
        """

        res = np.zeros((len(self.learners), len(self.query_strategies)))
        for enum_i, l in enumerate(self.learners):
            for enum_j, qs in enumerate(self.query_strategies):
                res[enum_i, enum_j] = self.heatmap[(l,qs)]

        qs_ordered = []
        for qs in all_qs_ordered:
            if qs in self.query_strategies:
                qs_ordered.append(qs)
        learners_ordered = []
        for l in all_learners_ordered:
            if l in self.learners:
                learners_ordered.append(l)


        # Define a custom colormap from light red to red
        greens = plt.cm.Greens
        reds = plt.cm.Reds
        blues = plt.cm.Blues
        purples = plt.cm.Purples

        # Normalize the data
        norm = Normalize(vmin=res.min(), vmax=res.max()/3)

        # Apply the custom colormap to the data
        red_colors = reds(norm(res))
        green_colors = greens(norm(res))
        blue_colors = blues(norm(res))
        purple_colors = purples(norm(res))



        # Erstellen des Plots
        fig, ax = plt.subplots(figsize=(1.5 * len(qs_ordered), 1.5 * len(learners_ordered)))

        # Zeichnen der Tafel mit den Farben basierend auf den Werten
        for i,l in enumerate(learners_ordered):
            for j,qs in enumerate(qs_ordered):
                if res[i,j] > 0:
                    ax.text(j + 0.5, i + 0.5, str(int(res[i,j])), ha='center', va='center', color='white',
                        fontsize=30,
                        weight='bold')
                else:
                    ax.text(j + 0.5, i + 0.5, str(int(res[i, j])), ha='center', va='center', color='black',
                            fontsize=30,
                            weight='bold')
                if qs in info:
                    colors = red_colors
                elif qs in repr:
                    colors = green_colors
                elif qs in hybr:
                    colors = blue_colors
                else:
                    colors = purple_colors
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=True, color=colors[i, j]))

        # adjust plot
        x_pos = np.arange(1, len(qs_ordered) + 1.5, 1) - .5
        y_pos = np.arange(1, len(learners_ordered) + 1.5, 1) - .5


        x_pos[-1] -= .5
        y_pos[-1] -= .5

        ax.set_xticks(x_pos)
        ax.set_yticks(y_pos)

        qs_names = [qs_for_figure[qs] for qs in qs_ordered]
        learner_names = [learner_for_figure[l] for l in learners_ordered]


        qs_names.append(" ")
        learner_names.append(" ")
        ax.set_xticklabels(qs_names, fontsize=40, rotation=45)
        ax.set_yticklabels(learner_names, fontsize=40)
        if self.statistical_significant:
            fig.suptitle("Setting: " + self.setting_name + ", Datasets: " + str(self.filter_ids),
                     fontsize=50)
            ax.set_title("(statistically significant)", fontsize=30)

        else:
            #fig.suptitle("Setting: " + self.setting_name + ", Datasets: " + str(self.filter_ids),
            #         fontsize=50, y=1.3)
            #ax.set_title("(not statistically significant)", fontsize=30)
            ax.set_title("Setting: " + self.setting_name + ", Datasets: " + str(self.filter_ids),
                     fontsize=50)


        if self.path_to_save is None:
            self.path_to_save = "FIGURES/HEATMAPS"
        PATH = self.path_to_save + "/"
        if not os.path.exists(PATH):
            os.makedirs(PATH)
        SAVE_PATH = PATH + self.setting_name +  "_AUBC.pdf"
        fig.savefig(SAVE_PATH, facecolor='white', transparent=True, bbox_inches='tight')
        if show_fig:
            plt.show()
        plt.close()
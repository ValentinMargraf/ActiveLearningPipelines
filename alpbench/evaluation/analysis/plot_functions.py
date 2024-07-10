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
            for enum,budget in enumerate(budgets):
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


    def show(self, show_fig = False):
        """
        This function plots the performance of different query strategies over the budget and saves
        it as .pdf under the specified path.
        """
        data = self.plot_data
        if len(data.keys()) == 0:
            return
        else:
            oid = self.openml_id
            # we want to order the QS by grouping them into categories, also each category gets a different
            # color coding (uncertainty based are redish, representative are greenish, hybrid are blueish,
            # random is pink)
            keys = data.keys()
            list_of_qs = list(keys)
            all_qs_ordered = ["random", "entropy", "least_confident", "margin", "power_margin",
                        "max_entropy", "bald", "power_bald", "qbc_variance_ratio",
                        "kmeans", "core_set","typ_cluster",
                        "cluster_margin", "falcun",  "weighted_cluster"]
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
                          "weighted_cluster": "turquoise", "falcun": "blue"}
            for cl, key in enumerate(ordered_qs):
                budget = np.array(data[key]['budget'])
                mu = np.array(data[key]['mean'])
                # std error
                std = np.array(data[key]['std'])/np.sqrt(self.num_seeds)
                qs = key
                cl = color_dict[qs]
                plt.plot(budget, mu, lw=2, label=key, color=cl)
                plt.fill_between(budget, mu+std, mu-std,  facecolor=cl, alpha=0.5)


            qs_for_figure = {"margin": "MS", "least_confident": "LC", "entropy": "ES",
                                "power_margin": "PowMS", "bald": "BALD", "power_bald": "PowBALD",
                                "max_entropy": "MaxEnt", "qbc_variance_ratio": "QBC VR",
                                "core_set": "CoreSet", "typ_cluster": "TypClu", "cluster_margin": "CluMS",
                                "weighted_cluster": "Clue", "falcun": "FALCUN", "random": "Rand"}
            learner_for_figure = {"knn_3": "KNN", "svm_rbf": "SVM (rbf)", "mlp": "MLP", "rf_entropy": "RF (entr)",
                                  "rf_gini": "RF (gini)", "svm_lin": "SVM (lin)", "catboost": "Catboost", "xgb": "XGB",
                                  "tabnet": "Tabnet", "tabpfn": "TabPFN"}

            handles, labels = plt.gca().get_legend_handles_labels()

            new_labels = []
            for label in labels:
                new_labels.append(qs_for_figure[label])

            plt.title(learner_for_figure[self.learner_name] + " on id " + str(self.openml_id), fontsize=30)
            plt.legend(handles, new_labels, fontsize=20, loc='lower right')# fontsize=25,loc='center left', bbox_to_anchor=(1, 0.5))
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
                self.path_to_save = "FIGURES/BUDGET_PERFORMANCE_PLOT/" +  str(self.openml_id) + "/"
                if not os.path.exists(self.path_to_save):
                    os.makedirs(self.path_to_save)
            self.path_to_save = self.path_to_save + str(self.learner_name) + ".pdf"
            fig.savefig(self.path_to_save, facecolor='white', transparent=True, bbox_inches='tight')
            if show_fig:
                plt.show()
            plt.close()
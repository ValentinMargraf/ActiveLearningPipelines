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

    This class plots the performance of different sampling strategies over the budget.

    Args:
        data (dict): The data to plot.
        path_to_save (str): The path to save the plot.

    Attributes:
        data (dict): The data to plot.
        path_to_save (str): The path to save the plot.

    """
    def __init__(self, data, path_to_save = None):
        self.data = data
        self.path_to_save = path_to_save

    def show(self):
        info = ["margin", "least_confident", "entropy", "power_margin", "bald", "power_bald",
         "max_entropy", "qbc_variance_ratio"]
        hybr = ["cluster_margin", "weighted_cluster","falcun"]
        repr = ["core_set",  "kmeans",  "typ_cluster"]
        rand = [ "random"]


        data = self.data["data"]
        # get first entry of first key
        # if dict is non empty
        if len(data.keys()) == 0:
            return
        else:
            oid = self.data["oid"]

            learner = list(data.keys())[0][0]

            setting_name = self.data['setting_name']
            metric_name = self.data['metric_name']
            keys = data.keys()
            list_of_keys = [key[-1] for key in keys]
            print("keys", list_of_keys)

            sampling_names = ["Rand", "MS",# "LC",
                              "ES",
                              #"QBC",
                                          # "MaxEnt",
                              "PowMS", #"BALD",
                              #"PowBALD",
                               "CoreSet",
                              "TypClu",
                              #"KMeans",
                              "CluMS",#
                                #,#"CLUE",
                              "FALCUN"
                              ]
            #keys[
            #    'cluster_margin', 'core_set', 'entropy', 'falcun', 'margin', 'power_bald', 'power_margin', 'random', 'typ_cluster']
            qstrats = ["random", "margin", "entropy", "power_margin", "power_bald", "core_set", "cluster_margin", "typ_cluster",
                       "falcun"]
            #subset = ["random", "margin", "core_set", "cluster_margin"]
            #qstrats = ["random", "entropy", "least_confident", "margin", "power_margin",
            #         "max_entropy", "bald", "power_bald", "qbc_variance_ratio",
            #         "kmeans", "core_set",
            #         "cluster_margin", "falcun", "typ_cluster", "weighted_cluster"]
            #order = [0,3,6,7,8,9,10,11,2,5,1,4,13,14,12]
            if len(qstrats) != len(list_of_keys):
                print("ERROR: qstrats and list_of_keys have different lengths")
                return
            order = [7, 4, 2, 6, 5, 1, 8, 0, 3]
            finetuned_order = []
            for enum,qs in enumerate(qstrats):
                if qs in qstrats:
                    finetuned_order.append(order[enum])
                else:
                    order = np.array(order) - 1
                    order = list(order)

            fig, ax = plt.subplots(1)
            # colormap of len(keys)
            #colors = plt.cm.viridis(np.linspace(0, 1, len(keys)))
            colors = plt.cm.tab20(np.linspace(0, 1, len(keys)))
            for cl, key in enumerate(keys):
                budget = np.array(data[key]['budget'])
                print("key", key)
                print("budget ", budget)
                mu = np.array(data[key]['mean'])
                std = np.array(data[key]['std'])
                method = key[-1]

                if method == "random":
                    cl = "magenta"
                if method == "least_confident":
                    cl = "rosybrown"
                elif method == "margin":
                    cl = "brown"
                elif method == "entropy":
                    cl = "orange"
                elif method == "power_margin":
                    cl = "red"
                elif method == "bald":
                    cl = "tomato"
                elif method == "power_bald":
                    cl = "coral"
                elif method == "max_entropy":
                    cl = "sandybrown"
                elif method == "qbc_variance_ratio":
                    cl = "peachpuff"
                repr_colors = ["limegreen", "forestgreen"]
                if  method == "core_set":
                    #cl ="limegreen"
                    cl = repr_colors[0]
                    #budget = budget[:length]
                    #budget[-1] = budget_val
                    #mu = mu[:length]
                    #std = std[:length]
                elif method == "typ_cluster":
                    cl = repr_colors[1]
                hybr_colors = ["indigo", "blue", "turquoise", "steelblue"]
                if method == "cluster_margin":
                    cl = "mediumblue"#hybr_colors[0]
                #elif method == "typ_cluster":
                #    cl = hybr_colors[1]
                elif method == "weighted_cluster":
                    cl = hybr_colors[3]
                elif method == "falcun":
                    cl = hybr_colors[2]


                if method in info:
                    plt.plot(budget, mu, lw=2, label=key[-1], color=cl)#, linestyle='--')
                    plt.fill_between(budget, mu+std, mu-std, alpha=0.5 ,facecolor=cl)
                elif method in hybr:
                    plt.plot(budget, mu, lw=2, label=key[-1], color=cl)#, linestyle='-.')
                    plt.fill_between(budget, mu+std, mu-std, alpha=0.5,  facecolor=cl,linestyle='--')
                elif method in repr:
                    plt.plot(budget, mu, lw=2, label=key[-1], color=cl)#, linestyle=':')
                    plt.fill_between(budget, mu+std, mu-std, alpha=0.5, facecolor=cl, linestyle=':')
                elif method in rand:
                    plt.plot(budget, mu, lw=2, label=key[-1], color=cl)#, linestyle=':')
                    plt.fill_between(budget, mu+std, mu-std,  facecolor=cl, alpha=0.5)


            if learner == "knn_3":
                learner_name = "KNN"
            elif learner == "svm_rbf":
                learner_name = "SVM"
            elif learner == "mlp":
                learner_name = "MLP"
            elif learner == "rf_entropy":
                learner_name = "RF"
            elif learner == "catboost":
                learner_name = "Catboost"
            elif learner == "xgb":
                learner_name = "XGB"
            elif learner == "tabnet":
                learner_name = "Tabnet"
            else:
                learner_name = "TabPFN"
            plt.title(learner_name + " on id " + str(oid), fontsize=30)
            #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            #plt.legend()
            handles, labels = plt.gca().get_legend_handles_labels()

            print("HANDLED", handles)
            print("labels", labels)
            #labels = [clums, core, mar, rand]

            #new_labels = ["Rand", "MS", "CoreSet", "CluMS"]
            #new_handles = [handles[3], handles[2], handles[1], handles[0]]

            # keys[
            #    'cluster_margin', 'core_set', 'entropy', 'falcun', 'power_bald', 'power_margin', 'random', 'typ_cluster']
            sampling_names = ['CluMS', 'CoreSet', 'ES', 'FALCUN', 'MS', 'PowBALD', 'PowMS', 'Rand', 'TypClu']
            #keys[
            #    'cluster_margin', 'core_set', 'entropy', 'falcun', 'margin', 'power_bald', 'power_margin', 'random', 'typ_cluster']

            # TODO print labels and convert to abbreviations
            new_handles=[]
            new_labels=[]
            for enum,idx in enumerate(order):
            #    #if qstrats[enum] in subset:
            #    #print(idx)
                new_handles.append(handles[idx])
                new_labels.append(sampling_names[idx])
            # set new labels


            #plt.legend(handles, labels,  fontsize=25,loc='center left', bbox_to_anchor=(1, 0.5))
            plt.legend(new_handles, new_labels,  fontsize=20,loc='center left', bbox_to_anchor=(1, 0.5))

            #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=20)
            plt.yticks(fontsize=20)
            plt.xticks(fontsize=20)



            ax.set_xlabel('Number of labeled instances', fontsize=25)
            ax.set_ylabel('test accuracy', fontsize=25)
            [l.set_visible(False) for (i, l) in enumerate(ax.xaxis.get_ticklabels()) if i % 2 != 0]

            # save image
            if self.path_to_save is not None:
                if not os.path.exists(self.path_to_save):
                    os.makedirs(self.path_to_save)
                self.path_to_save = self.path_to_save + setting_name + "_fixed_learner_" + metric_name + ".pdf"
                fig.savefig(self.path_to_save, facecolor='white', transparent=True, bbox_inches='tight')
            plt.close()
            #plt.show()





class HeatMapPlot:
    """HeatMapPlot

    This class plots a heatmap of the performance of different active learning pipelines as well as win or lose-
    matrices for each learner comparing different query strategies.

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
    def __init__(self, data, path_to_save = None, filter_ids = "all", take_statistical_insignificant=False):
        self.data = data
        self.path_to_save = path_to_save
        self.filter_ids = filter_ids
        self.take_statistical_insignificant = take_statistical_insignificant


    def show_heatmap(self):

        learners_ordered = ["knn_3", "svm_rbf", "rf_entropy", "catboost",
                            "xgb","mlp",
                            "tabnet",
                            "tabpfn"]
        info = ["margin", "entropy", "power_margin",  "power_bald"]
        repr = ["core_set", "typ_cluster"]
        hybr = ["cluster_margin", "falcun"]

        sampling_strategies_ordered = ["random" ,
                                       "margin",
                                       # "least_confident",
                                       "entropy", #"qbc_variance_ratio",
                                       #"max_entropy",
                                       "power_margin", #"bald",
                                       "power_bald",
                                       "core_set", #"kmeans",
                                       "typ_cluster",
                                       "cluster_margin",
                                       #"weighted_cluster",
                                       "falcun"
                                       ]



        learners = self.data['learners']
        sampling_strategies = self.data['sampling_strategies']
        heatmap = self.data['heatmap']


        res = np.zeros((len(learners_ordered), len(sampling_strategies_ordered)))
        for enum_i, l in enumerate(learners_ordered):
            for enum_j, qs in enumerate(sampling_strategies_ordered):
                res[enum_i, enum_j] = heatmap[(l,qs)]


        setting_name = self.data['setting_name']
        metric_name = self.data['metric_name'] + "_AUBC"


        # Define a custom colormap from light red to red
        light_red_to_red = LinearSegmentedColormap.from_list('light_red_to_red', ['lavender', 'darkblue'])
        greens = plt.cm.Greens
        reds = plt.cm.Reds
        blues = plt.cm.Blues
        purples = plt.cm.Purples

        # Normalize the data
        norm = Normalize(vmin=res.min(), vmax=res.max())

        # Apply the custom colormap to the data
        colors = light_red_to_red(norm(res))
        red_colors = reds(norm(res))
        green_colors = greens(norm(res))
        blue_colors = blues(norm(res))
        purple_colors = purples(norm(res))



        # Erstellen des Plots
        fig, ax = plt.subplots(figsize=(1.5 * len(sampling_strategies_ordered), 1.5 * len(learners_ordered)))

        # Zeichnen der Tafel mit den Farben basierend auf den Werten
        for i,l in enumerate(learners_ordered):
            for j,qs in enumerate(sampling_strategies_ordered):
                #print("hey", str(int(res_wins[i,j])) + "/" + str(int(res_losses[i,j])))
                #print("total " , int(res_wins[i,j] + res_losses[i,j]))
                ax.text(j + 0.5, i + 0.5, str(int(res[i,j])), ha='center', va='center', color='white',
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

        # Einstellungen für den Plot
        x_pos = np.arange(1, len(sampling_strategies_ordered) + 1.5, 1) - .5
        y_pos = np.arange(1, len(learners_ordered) + 1.5, 1) - .5


        x_pos[-1] -= .5
        y_pos[-1] -= .5

        ax.set_xticks(x_pos)
        ax.set_yticks(y_pos)

        learners_ordered = ["KNN", "SVM",  "RF", "Catboost", "XGB","MLP",
                            "Tabnet",
                            "TabPFN"]
        sampling_strategies_ordered = ["Rand",
                                       "MS", #"least_confident",
                                       "ES",  # "qbc_variance_ratio",
                                       # "max_entropy",
                                       "PowMS",  # "bald",
                                       "PowBALD",
                                       "CoreSet",
                                       "TypClu",
                                       # "kmeans",
                                       "CluMS",
                                       # "weighted_cluster",
                                       "FALCUN"
                                       ]

        sampling_names = list(sampling_strategies_ordered)
        learners_names = list(learners_ordered)

        sampling_names.append(" ")
        learners_names.append(" ")
        ax.set_xticklabels(sampling_names, fontsize=40, rotation=45)
        ax.set_yticklabels(learners_names, fontsize=40)
        if self.take_statistical_insignificant:
            fig.suptitle("Setting: " + setting_name + ", Datasets: " + str(self.filter_ids),
                     fontsize=50)
            ax.set_title("(statistically significant)", fontsize=30)

        else:
            fig.suptitle("Setting: " + setting_name + ", Datasets: " + str(self.filter_ids),
                     fontsize=50)
            ax.set_title("(not statistically significant)", fontsize=30)#, color='white')

        filter_ids = self.data['filter_ids']
        if self.path_to_save is not None:
            PATH = self.path_to_save + str(filter_ids) +"/"
            if not os.path.exists(PATH):
                os.makedirs(PATH)
            SAVE_PATH = PATH + setting_name +  metric_name +  "_"+str(filter_ids) +"_.pdf"
        print("saved in ", SAVE_PATH)
        fig.savefig(SAVE_PATH, facecolor='white', transparent=True, bbox_inches='tight')
        #plt.show()


    def show_win_matrix(self, ties=True):
        info = ["margin", "entropy", "power_margin",  "power_bald"]
        repr = ["core_set", "typ_cluster"]
        hybr = ["cluster_margin",  "falcun"]

        sampling_strategies_ordered = ["random" ,
                                       "margin",
                                       #"least_confident",
                                       "entropy", #"qbc_variance_ratio",
                                       #"max_entropy",
                                       "power_margin", #"bald",
                                       "power_bald",
                                       "core_set", #"kmeans",
                                       "typ_cluster",
                                       "cluster_margin",
                                       #"weighted_cluster",
                                       "falcun"]
        learners_ordered = ["knn_3", "svm_rbf", "mlp","rf_entropy","catboost", "xgb",
        "tabnet",  "tabpfn"]

        filter_ids = self.data['filter_ids']

        learners = self.data['learner'].keys()


        for learner in learners:
            #print("jo",learner)
            win_matrix = self.data['learner'][learner]
            print("WIN MATRIX",win_matrix)
            sampling_strategies = win_matrix.keys()


            # create numpy array from dict
            res_wins = np.zeros((len(sampling_strategies_ordered), len(sampling_strategies_ordered)))
            res_losses  = np.zeros((len(sampling_strategies_ordered), len(sampling_strategies_ordered)))

            for i, strat_1 in enumerate(sampling_strategies_ordered):
                for j, strat_2 in enumerate(sampling_strategies_ordered):
                    key = ((learner, strat_1), (learner, strat_2))

                    # check if key exists in win_matrix
                    if key in win_matrix.keys():
                        wins = win_matrix[key][0]
                        losses = win_matrix[key][1]
                    else:
                        wins = 0
                        losses = 0
                    if i == j:
                        wins = 0
                        losses = 0

                    res_wins[i, j] = wins
                    res_losses[i, j] = losses

            if ties:
                res_total = res_wins / (res_wins + res_losses)
            else:
                res_total = np.ones((len(sampling_strategies_ordered), len(sampling_strategies_ordered))) * self.data['num_dataset_ids']
                # set diag to 0
                np.fill_diagonal(res_total, 0)


            setting_name = self.data['setting_name']
            metric_name =  self.data['metric_name'] + "_AUBC"

            # create plot
            if ties:
                norm = Normalize(vmin=0, vmax=10)  # res_wins.min(), vmax=res_wins.max() * 1.5)
                colors = plt.cm.viridis(norm(10*res_total))
            else:
                light_red_to_red = LinearSegmentedColormap.from_list('light_red_to_red', ['lavender', 'darkblue'])

                greens = plt.cm.Greens
                reds = plt.cm.Reds
                blues = plt.cm.Blues
                purples = plt.cm.Purples


                # Normalize the data
                norm = Normalize(vmin=res_wins.min(), vmax=res_wins.max())

                # Apply the custom colormap to the data
                colors = light_red_to_red(norm(res_wins))

                # Apply the custom colormap to the data
                colors = light_red_to_red(norm(res_wins))
                red_colors = reds(norm(res_wins))
                green_colors = greens(norm(res_wins))
                blue_colors = blues(norm(res_wins))
                purple_colors = purples(norm(res_wins))

                #norm = Normalize(res_wins.min(), vmax=res_wins.max() * 1.5)
                #colors = plt.cm.viridis(norm(res_wins))
                #colors = plt.cm.Reds(norm(res_wins))
            #print("COOLS",colors)
            # Use the 'Reds' colormap for red tones
            #colors = plt.cm.Reds(norm(res_wins))

            # Erstellen des Plots
            fig, ax = plt.subplots(figsize=(1.5 * len(sampling_strategies_ordered), 1.5 * len(sampling_strategies_ordered)))

            # Zeichnen der Tafel mit den Farben basierend auf den Werten
            for i,l in enumerate(sampling_strategies_ordered):
                for j,qs in enumerate(sampling_strategies_ordered):
                    #print("hey", str(int(res_wins[i,j])) + "/" + str(int(res_losses[i,j])))
                    #print("total " , int(res_wins[i,j] + res_losses[i,j]))
                    if ties:
                        ax.text(j + 0.5, i + 0.5, str(int(res_wins[i,j])) + "/" + str(int(res_wins[i,j] + res_losses[i,j])), ha='center', va='center', color='white',
                                fontsize=25,
                                weight='bold')
                    else:
                        ax.text(j + 0.5, i + 0.5,
                                str(int(res_wins[i, j])) + "/" + str(int(res_total[i, j])),
                                ha='center', va='center', color='white',
                                fontsize=25,
                                weight='bold')
                    if l in info:
                        colors = red_colors
                    elif l in repr:
                        colors = green_colors
                    elif l in hybr:
                        colors = blue_colors
                    else:
                        colors = purple_colors

                    ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=True, color=colors[i, j]))

            # Einstellungen für den Plot
            x_pos = np.arange(1, len(sampling_strategies_ordered) + 1.5, 1) - .5
            y_pos = np.arange(1, len(sampling_strategies_ordered) + 1.5, 1) - .5


            x_pos[-1] -= .5
            y_pos[-1] -= .5
            print("x_pos", x_pos)
            print("y_pos", y_pos)

            ax.set_xticks(x_pos)
            ax.set_yticks(y_pos)

            sampling_names = ["Rand", "MS", "ES", "PowMS", "PowBALD", "CoreSet", "TypClu", "CluMS", "FALCUN"]
                              # "LC",
                              #"ES",  #"QBC",
                                          # "MaxEnt",
                              #"PowMS", #"BALD",
                              #"PowBALD",
                              # "CoreSet",
                              #"KMeans",
                              #"CluMS",
                              #"TypClu"#,"CLUE",
                              #"FALCUN"
                              #]
            #sampling_names = list(sampling_strategies_ordered)
            learners_names = list(learners)

            sampling_names.append(" ")

            learners_ordered = ["knn_3", "svm_rbf", "mlp", "rf_entropy", "catboost", "xgb",
                                "tabnet", "tabpfn"]

            if learner == "knn_3":
                learner_name = "KNN"
            elif learner == "svm_rbf":
                learner_name = "SVM"
            elif learner == "mlp":
                learner_name = "MLP"
            elif learner == "rf_entropy":
                learner_name = "RF"
            elif learner == "catboost":
                learner_name = "Catboost"
            elif learner == "xgb":
                learner_name = "XGB"
            elif learner == "tabnet":
                learner_name = "Tabnet"
            elif learner == "tabpfn":
                learner_name = "TabPFN"


            learners_names.append(" ")
            ax.set_xticklabels(sampling_names, fontsize=40, rotation=45)
            ax.set_yticklabels(sampling_names, fontsize=40)
            ax.set_title("Setting: " + setting_name + ", Learner: " + learner_name,
                         fontsize=50)
            SAVE_PATH = "fig " + str(ties) + ".pdf"


            if self.path_to_save is not None:
                PATH = self.path_to_save + "/" + str(filter_ids) +"/"
                if not os.path.exists(PATH):
                    os.makedirs(PATH)
                SAVE_PATH = PATH + setting_name +  metric_name + "_"+str(learner) + ".pdf"
                print("SAVE PATH", SAVE_PATH)
                fig.savefig(SAVE_PATH, facecolor='white', transparent=True, bbox_inches='tight')
            #plt.show()

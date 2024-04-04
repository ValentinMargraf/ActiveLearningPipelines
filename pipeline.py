import pandas as pd
import numpy as np
import openml
from sklearn import compose, impute, metrics, pipeline, preprocessing, tree, decomposition
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import pairwise_distances, make_scorer, roc_auc_score, silhouette_score, pairwise_distances
from scipy.spatial.distance import pdist
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression as lr
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.svm import SVC as svm
import sklearn.preprocessing as preprocessing
import sklearn.impute as impute
from openml import config, study, tasks, runs, extensions
from sklearn.naive_bayes import MultinomialNB as mnb
from sklearn.ensemble import ExtraTreesClassifier as etc
from sklearn.neighbors import KNeighborsClassifier as nn
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from scipy.stats import entropy
from uncertainty_quantifier import RandomForestEns as ens
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import scipy
from pymfe.mfe import MFE


class ALPipeline():

    def load_meta_data(self):

        self.num_attributes = self.X.shape[1]
        self.num_classes = self.num_classes
        self.pool_size = len(self.yu)
        means_per_attribute = np.mean(self.Xu, axis=0)
        stds_per_attribute = np.std(self.Xu, axis=0)

        self.mean = np.array([np.min(means_per_attribute), np.mean(means_per_attribute), np.max(means_per_attribute)])
        self.std = np.array([np.min(stds_per_attribute), np.mean(stds_per_attribute), np.max(stds_per_attribute)])
        """
        # compute "aleatoric uncertainty of the dataset"
        RF_ensemble = ens()
        RF_ensemble.fit_ensemble(self.Xl, self.yl)
        #print("shap yl", self.yl)
        #print("openmlid ", self.data[0])
        probs = RF_ensemble.predict_proba(self.Xu)
        #print("probv shape", probs.shape)
        #print("log ", np.log2(probs.shape[1]))
        # only take instances within Xu that are close to the decision boundary of lr
        a_u = RF_ensemble._aleatoric_uncertainty_entropy(probs)
        e_u = RF_ensemble._epistemic_uncertainty_entropy(probs)
        self.aleatoric_uncertainty = np.array([np.min(a_u), np.mean(a_u), np.max(a_u), np.sum(a_u)])
        self.epistemic_uncertainty = np.array([np.min(e_u), np.mean(e_u), np.max(e_u), np.sum(e_u)])

        # only close to decision boundary
        close_to_boundary = []
        probs_mean = probs.mean(axis=-1)
        for i, prob in enumerate(probs_mean):
            try:
                if np.max(prob) - np.sort(prob)[-2] < .25:
                    close_to_boundary.append(i)
            except IndexError as ie:
                close_to_boundary.append(i)
        #print("close to boundary", close_to_boundary)
        if len(close_to_boundary) > 0:
            #print("shape of probs", probs.shape)
            close_to_boundary = np.array(close_to_boundary).astype(int)
            probs_subset = RF_ensemble.predict_proba(self.Xu[close_to_boundary])
            a_u_subset = RF_ensemble._aleatoric_uncertainty_entropy(probs_subset)
            e_u_subset = RF_ensemble._epistemic_uncertainty_entropy(probs_subset)
            self.aleatoric_uncertainty_subset = np.array([np.min(a_u_subset), np.mean(a_u_subset), np.max(a_u_subset), np.sum(a_u_subset)])
            self.epistemic_uncertainty_subset = np.array([np.min(e_u_subset), np.mean(e_u_subset), np.max(e_u_subset), np.sum(e_u_subset)])
        else:
            self.aleatoric_uncertainty_subset = np.array([0, 0, 0, 0])
            self.epistemic_uncertainty_subset = np.array([0, 0, 0, 0])
        """
        # performance based meta features (k-fold cross validation)
        knn1 = nn(n_neighbors=1)
        dt1 = tree.DecisionTreeClassifier(max_depth=1)
        nb = GaussianNB()
        rf1 = rf(n_estimators=100)
        models = [knn1, dt1, nb, rf1]
        num_folds = 5
        accs = {}
        aucs = {}
        for model in models:
            accs[model] = []
            aucs[model] = []
        # Define your multiclass scoring function using ROC AUC with 'ovr'
        # this may give an index error if there is only one class in the data, therefore catch the error
        #print("DATATYPE ", type(accs[model]))
        #folds = KFold(n_splits=num_folds)

        folds = KFold(n_splits=num_folds, random_state=self.seed, shuffle=True)
        for i, (train_index, test_index) in enumerate(folds.split(self.Xl, self.yl)):
            #if len(np.unique(self.yl[train_index])) == len(np.unique(self.yl[test_index])) == self.num_classes:

            for model in models:
                model.fit(self.Xl[train_index], self.yl[train_index])
                preds = model.predict(self.Xl[test_index])
                auc_yl = np.zeros((len(self.yl[test_index]), self.num_classes))
                for i in range(len(self.yl[test_index])):
                    auc_yl[i, int(self.yl[test_index][i])] = 1
                auc_preds = np.zeros((len(preds), self.num_classes))
                for i in range(len(preds)):
                    auc_preds[i, int(preds[i])] = 1
                acc = metrics.accuracy_score(self.yl[test_index], preds)
                try:
                    auc = roc_auc_score(auc_yl, auc_preds, multi_class='ovr')
                except ValueError as ve:
                    auc = 0

                accs[model].append(acc)
                aucs[model].append(auc)
            self.accs = []
            self.aucs = []
            for model in models:
                self.accs.append(np.mean(np.array(accs[model])))
                self.aucs.append(np.mean(np.array(aucs[model])))
            self.accs = np.array(self.accs)
            self.aucs = np.array(self.aucs)

        raw_Xl = self.Xl.flatten().reshape(-1, 1)
        raw_Xu = self.Xu.flatten().reshape(-1, 1)
        raw_yl = self.yl.flatten().reshape(-1, 1)


        self.raw_data = np.concatenate([raw_Xl, raw_Xu, raw_yl], axis=0).flatten()
        self.class_imbalance = scipy.stats.entropy((np.bincount(self.yl.astype(int)) / len(self.yl)), base=self.num_classes)
        """
        # put all meta features into a vector
        mfe = MFE(groups="all")
        mfe.fit(self.Xl, self.yl)
        ft = mfe.extract()
        length = len(ft) // 2
        self.pymfe_all_labeled = np.array(ft[-length:]).flatten()

        mfe = MFE(groups="all")
        mfe.fit(self.Xu)
        ft = mfe.extract()
        length = len(ft) // 2
        self.pymfe_all_unlabeled = np.array(ft[-length:]).flatten()
        """
        self.concatenated = ([self.num_attributes,
                            self.num_classes,
                            self.pool_size,
                            self.class_imbalance,
                            self.mean,
                            self.std,
                            #self.aleatoric_uncertainty,
                            #self.epistemic_uncertainty,
                            #self.aleatoric_uncertainty_subset,
                            #self.epistemic_uncertainty_subset,
                            self.accs,
                            self.aucs])
                             #,)
                            #self.pymfe_all_labeled,
                            #self.pymfe_all_unlabeled]

        self.meta_features = np.concatenate(
            [np.round(np.ravel(item),5) if isinstance(item, np.ndarray) else [np.round(item,5)] for item in self.concatenated])

        # convert all entries to float
        #self.meta_features = self.meta_features.astype(float)



    def plot(self, string_to_save="None"):

        # Plotting the decision boundary along with the dataset
        X = self.X
        y = self.y
        X_u = self.Xu
        y_u = self.yu
        X_l = self.Xl
        y_l = self.yl
        X_te = self.Xte
        y_te = self.yte

        # Creating a meshgrid to plot the decision boundary
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                             np.arange(y_min, y_max, 0.01))

        # Predicting the labels for each point in the mesh
        Z = self.learner.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # defining custom cmap
        # Defining two specific colors
        color1 = 'paleturquoise'
        color2 = 'antiquewhite'
        color3 = 'palegreen'
        # Creating a custom colormap with these two colors
        colors = [color1, color2, color3] if self.num_classes == 3 else [color1, color2]
        n_bins = 3  if self.num_classes == 3 else 2
        cmap_name = 'my_list'
        cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

        # Plotting the decision boundary along with the dataset
        figure = plt.figure(figsize=(4, 2))
        plt.contourf(xx, yy, Z, alpha=0.8, cmap=cm)

        #plt.scatter(X_te[:, 0][y_te == 0], X_te[:, 1][y_te == 0], marker=0, color="steelblue", label="test - 0", s=50)
        #plt.scatter(X_te[:, 0][y_te == 1], X_te[:, 1][y_te == 1], marker=0, color="goldenrod", label="test - 1", s=50)
        #if self.num_classes==3:
        #    plt.scatter(X_te[:, 0][y_te == 2], X_te[:, 1][y_te == 2], marker=".", color="green", label="unlabelled - 2")
        #teal
        #coral
        #lightgreen
        plt.scatter(X_u[:, 0][y_u == 0], X_u[:, 1][y_u == 0], marker="D", facecolors='none', color="blue", label="unlabelled - 0", s=50)
        plt.scatter(X_u[:, 0][y_u == 1], X_u[:, 1][y_u == 1], marker="D", facecolors='none',color="orange", label="unlabelled - 1", s=50)
        if self.num_classes==3:
            plt.scatter(X_u[:, 0][y_u == 2], X_u[:, 1][y_u == 2], marker="D", facecolors='none', color="darkgreen", label="unlabelled - 2",
                    s=50)

        plt.scatter(X_l[:, 0][y_l == 0], X_l[:, 1][y_l == 0], marker="D", color="blue", label="labelled - 0", s=80)
        plt.scatter(X_l[:, 0][y_l == 1], X_l[:, 1][y_l == 1], marker="D", color="orange", label="labelled - 1", s=80)
        if self.num_classes==3:
            plt.scatter(X_l[:, 0][y_l == 2], X_l[:, 1][y_l == 2], marker="D", color="darkgreen", label="labelled - 1", s=80)


        string = "Accuracy: " + str(np.round(self.current_test_metrics[0], 3))

        plt.title(string, fontsize=30)
        plt.xticks([])
        plt.yticks([])
        # plt.xlabel('Feature 1')
        # plt.ylabel('Feature 2')
        # plt.legend(loc='lower right', bbox_to_anchor=(1.2, 0.5))
        # plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.13), ncol=6)  # Adjust the bbox_to_anchor for positioning
        figure.savefig('figs/'+str(string_to_save)+'.pdf', facecolor='white', bbox_inches='tight', transparent=True)
        #return figure



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

    def __init__(self, data, learner,learner_name,  sampling_approach, preprocessing_PCA = False, num_to_label = 10, synthetic=False, iterations=10, initial_size=30):
        """
        This class implements the active learning pipeline. It is constructed from a dataset, a learner and a sampling approach.

        :param data: tuple of (openmlid, split, seed)
        :param learner: sklearn learner
        :param sampling_approach: string, one of ['random', 'cluster_based', 'cluster_and_diversity_based', 'least_confident', 'margin_sampling', 'entropy', 'qbc']
        :param preprocessing_PCA: bool, whether to preprocess the data with PCA
        :param num_to_label: int, number of samples to label in each iteration
        :param synthetic: bool, whether to use synthetic data or real data
        """
        self.data = data
        self.learner = learner
        self.learner_name = learner_name
        self.sampling_approach = sampling_approach
        self.preprocessing_PCA = preprocessing_PCA
        self.num_to_label = num_to_label
        self.imputer = impute.SimpleImputer(missing_values=np.nan, strategy='mean')
        self.encoder = preprocessing.OrdinalEncoder()
        self.X = None
        self.y = None
        self.total_num_samples = None
        self.Xl = None
        self.yl = None
        self.Xu = None
        self.yu = None
        self.Xte = None
        self.yte = None
        self.budget = None
        self.num_classes = None
        self.synthetic = synthetic
        self.iterations = iterations
        self.initial_size = initial_size

        if not self.synthetic:
            self.load_split_data(data)
            self.num_X = len(self.X)
            self.num_Xl = len(self.Xl)
        self.current_test_metrics = None
        self.name = None
        self.sample = None
        self.current_test_acc = None
        self.seed = data[1]






    def load_split_data(self, data):
        """
        Loads the data from openml and splits it into train/test and labeled/unlabeled

        :param data: tuple of (openmlid, split, seed)
        :return: None
        """

        openmlid, seed  = data
        try:
            ds = openml.datasets.get_dataset(openmlid)
            # print("dataset info loaded")
            df = ds.get_data()[0]
            num_rows = len(df)

            # print("Data in memory, now creating X and y")

            # prepare label column as numpy array
            X = np.array(df.drop(columns=[ds.default_target_attribute]).values)
            y = np.array(df[ds.default_target_attribute].values)
            if y.dtype != int:
                y_int = np.zeros(len(y)).astype(int)
                vals = np.unique(y)
                for i, val in enumerate(vals):
                    mask = y == val
                    y_int[mask] = i
                y = y_int

        except Exception as e:
            task = openml.tasks.get_task(openmlid)
            X, y = task.get_X_and_y()

        X  = self.preprocess_data(X)


        # random shuffle
        arr = np.arange(y.shape[0])

        np.random.seed(seed)
        np.random.shuffle(arr)
        X = X[arr]
        y = y[arr]
        #print("X", X)
        #print("y", y)

        self.X = X
        self.y = y
        self.total_num_samples = len(y)
        self.num_classes = len(np.unique(y))
        # split into train/test
        train_X, X_te, train_labels, y_te = train_test_split(X, y, test_size=1 / 3, random_state=seed, stratify=None)
        # split train further into D_L, D_U
        # shuffle train_X and train_labels

        """
        arr = np.arange(len(train_labels))
        np.random.seed(seed)
        np.random.shuffle(arr)
        train_X = train_X[arr]
        train_labels = train_labels[arr]


        split = 300 / len(train_labels)

        # get minor class
        minor_class = np.argmin(np.bincount(train_labels.astype(int)))
        # get ratio of minor class to all other classes
        ratios = [np.sum(train_labels == i) / np.sum(train_labels == minor_class) for i in range(self.num_classes)]
        #print("ratios", ratios)
        # sample 5 instances from minor class, from the other according to the ratio
        X_l = []
        y_l = []
        X_u = []
        y_u = []
        for i in range(self.num_classes):
            X_l.append(train_X[train_labels == i][:int(5 * ratios[i])])
            y_l.append(train_labels[train_labels == i][:int(5 * ratios[i])])
            X_u.append(train_X[train_labels == i][int(5 * ratios[i]):])
            y_u.append(train_labels[train_labels == i][int(5 * ratios[i]):])
        X_l = np.concatenate(X_l, axis=0)
        y_l = np.concatenate(y_l, axis=0)
        X_u = np.concatenate(X_u, axis=0)
        y_u = np.concatenate(y_u, axis=0)
        """
        #print("SPLIT" , split)
        #split = 5 * self.num_classes / len(train_labels)

        split = self.initial_size / len(train_labels)
        print(self.initial_size)
        print(len(train_labels))
        print("split", split)
        X_u, X_l, y_u, y_l = train_test_split(train_X, train_labels, test_size=split, random_state=seed, stratify=None)

        #print("\nclass imbalance", np.bincount(y))
        #print("class imbalance labeled", np.bincount(y_l))

        self.Xl = X_l
        self.yl = y_l
        self.Xu = X_u
        self.yu = y_u
        self.Xte = X_te
        self.yte = y_te
        self.budget = min(200, len(self.yu)/2)
        #print("num yl", len(self.yl))
        #print("budget", self.budget)

    def preprocess_data(self, X):
        """
        Preprocesses the data by imputing missing values and encoding categorical features

        :param X: numpy array of shape (num_samples, num_features)
        :return: numpy array of shape (num_samples, num_features)
        """
        X = self.encoder.fit_transform(X)
        X = self.imputer.fit_transform(X)

        if self.preprocessing_PCA:
            # do pca
            pca = decomposition.PCA(n_components=2)
            X = pca.fit_transform(X)
            # take first two components
            #X = X[:,:2]
        return X

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


    def fit(self):
        """
        Fits the learner on the labeled data
        :return: None
        """

        self.learner.fit(self.Xl, self.yl)

    def predict(self, X):
        return self.learner.predict(X)

    def get_oracle_ids(self):
        # RANDOM
        if self.sampling_approach == 'random':
            return np.random.choice(np.arange(len(self.Xu)), self.num_to_label, replace=False)
        elif self.sampling_approach == 'least_confident':
            clf = self.learner
            clf.fit(self.Xl, self.yl)
            probas = clf.predict_proba(self.Xu)
            probas_max = probas.max(axis=-1)
            least_confident_ids = np.argsort(probas_max)[:self.num_to_label]
            return least_confident_ids
        elif self.sampling_approach == 'margin_sampling' or self.sampling_approach == 'margin_sampling_independent':
            clf = self.learner if self.sampling_approach == 'margin_sampling' else lr()
            clf.fit(self.Xl, self.yl)
            probas = clf.predict_proba(self.Xu)
            # get ids where predicted probabilities for two different classes are closest
            margins = []
            for i in range(len(probas)):
                # most likely
                sorted = np.sort(probas[i])
                most_likely_prob = sorted[-1]
                second_most_likely_prob = sorted[-2]
                margins.append(most_likely_prob - second_most_likely_prob)
            margins = np.array(margins)

            margin_ids = np.argsort(margins)[:self.num_to_label]
            #print("margins",margin_ids)
            return margin_ids
        elif self.sampling_approach == 'power_margin_sampling' or self.sampling_approach == 'power_margin_sampling_independent':
            clf = self.learner if self.sampling_approach == 'power_margin_sampling' else lr()
            clf.fit(self.Xl, self.yl)
            probas = clf.predict_proba(self.Xu)
            # get ids where predicted probabilities for two different classes are closest
            margins = []
            for i in range(len(probas)):
                # most likely
                sorted = np.sort(probas[i])
                most_likely_prob = sorted[-1]
                second_most_likely_prob = sorted[-2]
                margins.append(most_likely_prob - second_most_likely_prob)
            margins = np.array(margins)
            # power transform
            np.random.seed(self.seed)
            margins = np.log(margins + 1e-8) + np.random.gumbel(size=len(margins))
            margins = 1 - margins
            margin_ids = np.argsort(margins)[-self.num_to_label:]
            #print(margin_ids)
            return margin_ids

        elif self.sampling_approach == 'weighted_cluster' or self.sampling_approach == 'weighted_cluster_independent':
            clf = self.learner if self.sampling_approach == 'weighted_cluster' else lr()
            clf.fit(self.Xl, self.yl)
            scores = clf.predict_proba(self.Xu)  + 1e-8
            entropy = -np.sum(scores * np.log(scores), axis=1)
            from sklearn.cluster import KMeans
            from sklearn.metrics.pairwise import euclidean_distances

            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                kmeans = KMeans(n_clusters=self.num_classes)
                kmeans.fit(self.Xu, sample_weight=entropy)
                class_ids = [np.argwhere(kmeans.labels_ == i) for i in range(self.num_classes)]
                centers = kmeans.cluster_centers_
                dists = euclidean_distances(centers, self.Xu)
                sort_idxs = dists.argsort(axis=1)
                q_idxs = []
                n = len(self.Xu)
                # taken from https://github.com/virajprabhu/CLUE/blob/main/CLUE.py
                ax, rem = 0, n
                idxs_unlabeled = np.arange(n)
                while rem > 0:
                    q_idxs.extend(list(sort_idxs[:, ax][:rem]))
                    q_idxs = list(set(q_idxs))
                    rem = n - len(q_idxs)
                    ax += 1
                return idxs_unlabeled[q_idxs[:self.num_to_label]]

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


    def init(self):
        self.fit()
        preds = self.predict(self.Xte)
        test_acc = metrics.accuracy_score(self.yte, preds)
        # convert to array of shape (num_samples, num_classes)
        auc_te = np.zeros((len(self.yte), self.num_classes))
        for i in range(len(self.yte)):
            auc_te[i, int(self.yte[i])] = 1
        auc_preds = np.zeros((len(preds), self.num_classes))
        for i in range(len(preds)):
            auc_preds[i, int(preds[i])] = 1
        try:
            test_auc = metrics.roc_auc_score(auc_te, auc_preds, average='macro', multi_class='ovr')
        except Exception as e:
            #print(f"An error occurred: {e}")
            test_auc = None
        test_prec = metrics.precision_score(auc_te, auc_preds, average='macro', zero_division=np.nan)
        test_rec = metrics.recall_score(auc_te, auc_preds, average='macro', zero_division=np.nan)
        test_f1 = metrics.f1_score(auc_te, auc_preds, average='macro')
        self.current_test_metrics = [test_acc, test_auc, test_prec, test_rec]
        self.num_Xl = len(self.Xl)
        self.num_X = len(self.X)

    def update(self):
        ids = self.get_oracle_ids()
        self.Xl = np.concatenate([self.Xl, self.Xu[ids]])
        self.yl = np.concatenate([self.yl, self.yu[ids]])
        self.Xu = np.delete(self.Xu, ids, axis=0)
        self.yu = np.delete(self.yu, ids, axis=0)
        self.fit()
        preds = self.predict(self.Xte)
        test_acc = metrics.accuracy_score(self.yte, preds)
        # convert to array of shape (num_samples, num_classes)
        auc_te = np.zeros((len(self.yte), self.num_classes))
        for i in range(len(self.yte)):
            auc_te[i, int(self.yte[i])] = 1
        auc_preds = np.zeros((len(preds), self.num_classes))
        for i in range(len(preds)):
            auc_preds[i, int(preds[i])] = 1
        try:
            test_auc = metrics.roc_auc_score(auc_te, auc_preds, average='macro', multi_class='ovr')
        except Exception as e:
            #print(f"An error occurred: {e}")
            test_auc = None
        test_prec = metrics.precision_score(auc_te, auc_preds, average='macro', zero_division=np.nan)
        test_rec = metrics.recall_score(auc_te, auc_preds, average='macro', zero_division=np.nan)
        #test_f1 = metrics.f1_score(auc_te, auc_preds, average='macro')
        self.current_test_metrics = [test_acc, test_auc, test_prec, test_rec]
        self.num_Xl = len(self.Xl)




def run_active_pipeline(openmlid, seed, learner, sampling_approach, num_to_label = 10, result_processor=None, df = None, iterations=10, initial_size=30):


    import warnings
    from sklearn.exceptions import ConvergenceWarning

    # Suppress FutureWarning for the specified category
    warnings.simplefilter(action='ignore', category=FutureWarning)
    # Filter out ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    data = (openmlid, seed)

    svm_lin = svm(kernel='linear', probability=True)
    svm_rbf = svm(kernel='rbf', probability=True)
    rf1 = rf(n_estimators=100, max_depth=10, criterion='entropy')
    rf2 = rf(n_estimators=100, max_depth=10, criterion='gini')
    knn = nn(n_neighbors=5)
    lr1 = lr()
    mnb1 = mnb()
    etc1 = etc(n_estimators=100, max_depth=10, criterion='entropy')
    etc2 = etc(n_estimators=100, max_depth=10, criterion='gini')
    nb = GaussianNB()
    mlp = MLPClassifier()
    learner_name = learner
    if learner == 'svm_lin':
        learner = svm_lin
    elif learner == 'rf1':
        learner = rf1
    elif learner == 'lr1':
        learner = lr1
    elif learner == 'etc1':
        learner = etc1


    autoAL = ALPipeline(data, learner, learner_name, sampling_approach,  preprocessing_PCA=False, num_to_label=num_to_label,synthetic=False, iterations=iterations, initial_size=initial_size)
    autoAL.init()
    init_test_metrics = autoAL.current_test_metrics

    # meta features computen und loggen!
    #autoAL.load_meta_data()
    #meta_feats = autoAL.meta_features
    #print("meta feta shape", meta_feats.shape)

    #total_chars = sum(len(str(s)) for s in meta_feats)

    YES = False

    #print("Total number of characters:", total_chars)    # if result_processor is not None:
    if result_processor is not None:

        result_processor.process_logs({
            'labeling_log': {
                'iteration': 0,
                'num_Xl': autoAL.num_Xl,
                'num_X': autoAL.num_X#,
                #'meta_feats': str(autoAL.meta_features)
            }
        })
        result_processor.process_logs({
            'accuracy_log': {
                'iteration': 0,
                'test_acc': init_test_metrics[0],
                'test_auc': init_test_metrics[1],
                'test_prec': init_test_metrics[2],
                'test_rec': init_test_metrics[3]
            }
        })



    #num_iterations = int(autoAL.budget // autoAL.num_to_label)
    #for iter_ in range(1, num_iterations + 1):
    for iter_ in range(autoAL.iterations):
        autoAL.update()
        current_test_metrics = autoAL.current_test_metrics
        #print("current_test_metrics", current_test_metrics)
        # meta features computen und loggen!
        #autoAL.load_meta_data()
        #meta_feats = autoAL.meta_features
        #total_chars = sum(len(str(s)) for s in meta_feats)
        #print("Total number of characters:", total_chars)  # if result_processor is not None:

        # if result_processor is not None:
        if result_processor is not None:

            result_processor.process_logs({
                'labeling_log': {
                    'iteration': iter_,
                    'num_Xl': autoAL.num_Xl,
                    'num_X': autoAL.num_X#,
                    #'meta_feats': str(autoAL.meta_features)
                }
            })
            result_processor.process_logs({
                'accuracy_log': {
                    'iteration': iter_,
                    'test_acc': current_test_metrics[0],
                    'test_auc': current_test_metrics[1],
                    'test_prec': current_test_metrics[2],
                    'test_rec': current_test_metrics[3]
                }
            })





def main():
    kickout_ids = []
    for ID in [10]:#3,6,8,10,11,12,14,15,16,18,20,21,22,23,26,28,29,30,31,32,36,37,39,40,41,43,44,46,48,49,50,53,54,59,60,61,62,151,155,161,162,164,180,181,182,183,184,187,189,197,209,223,225,227,230,279,285,287,292,294,300,307,312,313,329,333,334,335,336,337,338,375,377,383,384,385,386,387,388,389,391,392,394,395,397,398,400,401,444,446,448,458,461,463,464,469,475,478,679,685,694,714,715,716,717,718,719,720,721,722,723,724,725,726,727,728,730,732,733,734,735,736,737,740,741,742,743,744,745,746,747,748,749,750,751,752,753,754,756,761,762,763,766,768,769,770,771,772,773,774,775,776,778,779,782,783,784,788,789,792,793,794,795,796,797,799,801,803,805,806,807,808,811,812,813,814,816,818,819,820,821,822,823,824,825,826,827,828,829,830,832,833,834,837,838,841,843,845,846,847,849,850,851,853,855,860,863,865,866,867,868,869,870,871,872,873,875,876,877,878,879,880,881,884,885,886,888,889,890,895,896,900,901,902,903,904,906,907,908,909,910,911,912,913,914,915,916,917,918,920,921,922,923,924,925,926,931,932,933,934,935,936,937,941,943,948,949,950,952,953,954,955,956,958,959,962,965,969,970,971,973,974,976,978,979,980,983,987,991,994,995,996,997,1004,1005,1006,1011,1012,1013,1014,1016,1019,1020,1021,1022,1025,1026,1036,1038,1040,1041,1043,1044,1045,1046,1048,1049,1050,1054,1059,1061,1063,1064,1065,1066,1067,1071,1073,1075,1077,1078,1080,1084,1100,1106,1115,1116,1120,1121,1122,1123,1124,1125,1126,1127,1129,1131,1132,1133,1135,1136,1137,1140,1141,1143,1144,1145,1147,1148,1149,1150,1151,1152,1153,1154,1155,1156,1157,1158,1159,1160,1162,1163,1164,1165,1167,1169,1217,1236,1237,1238,1412,1413,1441,1442,1443,1444,1446,1447,1448,1449,1450,1451,1453,1454,1455,1457,1459,1460,1464,1467,1471,1472,1473,1475,1481,1482,1483,1486,1487,1488,1489,1496,1498,1500,1501,1503,1505,1507,1508,1509,1510,1512,1513,1519,1520,1527,1528,1529,1531,1533,1534,1535,1537,1538,1539,1540,1542,1543,1545,1546,1556,1557,1565,1568,1600,3560,3902,3904,3913,3917,4134,4135,4153,4340,4534,4538,9976,9977,9978,9981,9985,10093,10101,14952,14954,14965,14969,14970,40474,40475,40476,40477,40478,125920,125922,146195,146800,146817,146819,146821,146822,146824,146825,167119,167120,167124,167125,167140,167141]:

        for seed in [0]:#np.arange(30):
            #run_active_pipeline(openmlid=ID, seed=seed, learner='rf1', sampling_approach='weighted_cluster',num_to_label=1, result_processor=None, iterations=10, initial_size=30)
            data = (ID, seed)
            rf1 = rf(n_estimators=100, max_depth=10, criterion='entropy')
            etc1 = etc(n_estimators=100, max_depth=10, criterion='entropy')
            autoAL = ALPipeline(data, etc1, 'etc1', 'weighted_cluster', preprocessing_PCA=False, num_to_label=5, synthetic=False, iterations=10, initial_size=100)


            num_classes = len(np.unique(autoAL.yl))
            if num_classes < 2:
                kickout_ids.append(ID)
                print("ID", ID, "only has one class")
                break
    print("kickout", kickout_ids)


if __name__ == '__main__':
    main()


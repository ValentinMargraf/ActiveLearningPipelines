import pandas as pd
import numpy as np
import openml
from openml import config, study, tasks, runs, extensions
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn import metrics
from scipy import stats
from uncertainty_quantifier import RandomForestEns as RFEns

class ActiveLearningPipeline():

    def __init__(self, scenario_id, scenario_name, openmlid, seed, setting_id, init_query, learner, query_strategy, query_learner, num_to_label, iterations):
        """
        This class implements the active learning pipeline. It is constructed from a dataset (openmlid + seed, which determines the initial
        split into labeled/unlabeled/test dataset), a setting (empty, small, medium, large) an initial query strategy
        (only relevant, if setting = empty), a learner, a query strategy, a learner for the query strategy.
        """
        self.scenario_id = scenario_id
        self.scenario_name = scenario_name
        self.openmlid = openmlid
        self.seed = seed

        self.setting_id = setting_id
        self.init_query = init_query
        self.learner = learner
        self.query_strategy = query_strategy
        self.query_learner = query_learner

        self.num_to_label = num_to_label
        self.iterations = iterations

        self.X = None
        self.y = None
        self.Xl = None
        self.yl = None
        self.Xu = None
        self.yu = None
        self.Xtest = None
        self.ytest = None
        self.num_Xl = None
        self.num_X = None
        self.indices_labeled = None
        self.indices_unlabeled = None
        self.indices_test = None

        self.current_test_metrics = None

        if self.setting == "empty":
            self.initial_size = 0
        elif self.setting == "small":
            self.initial_size = 30
        elif self.setting == "medium":
            self.initial_size = 100
        elif self.setting == "large":
            self.initial_size = 300

        self.num_classes = None

    def get_data(self):
        """
        This method retrieves the data from OpenML.
        """
        try:
            ds = openml.datasets.get_dataset(openmlid)
            # print("dataset info loaded")
            df = ds.get_data()[0]
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

        self.X = X
        self.y = y



    def get_split_indices(self):
        """
        This method splits the data into a training and test set.
        """
        self.indices_train, self.indices_test = train_test_split(np.arange(self.X.shape[0]), test_size=1/3, random_state=self.seed)
        # depending on the setting, calculate the split
        if self.setting != "empty":
            split = self.initial_size / len(self.indices_train)
            self.indices_labeled, self.indices_unlabeled = train_test_split(self.indices_train, test_size=split, random_state=self.seed)
        else:
            self.indices_labeled = []
            self.indices_unlabeled = self.indices_train

    def get_split_data(self):
        """
        This method splits the data into a training and test set.
        """
        self.Xtest = self.X[self.indices_test]
        self.ytest = self.y[self.indices_test]
        self.Xl = self.X[self.indices_labeled]
        self.yl = self.y[self.indices_labeled]
        self.Xu = self.X[self.indices_unlabeled]
        self.yu = self.y[self.indices_unlabeled]
        self.num_Xl = len(self.Xl)
        self.num_X = len(self.X)

    def fit(self):
        """
        Fits the learner on the labeled data
        :return: None
        """

        self.learner.fit(self.Xl, self.yl)

    def predict(self, X):
        return self.learner.predict(X)


    def get_oracle_ids(self):
        # TODO CHECK FOR ERRORS
        # RANDOM
        if self.sampling_approach == 'random':
            np.random.seed(self.seed)
            return np.random.choice(np.arange(len(self.Xu)), self.num_to_label, replace=False)
        elif self.sampling_approach == 'least_confident':
            clf = self.query_learner
            clf.fit(self.Xl, self.yl)
            probas = clf.predict_proba(self.Xu)
            probas_max = probas.max(axis=-1)
            least_confident_ids = np.argsort(probas_max)[:self.num_to_label]
            return least_confident_ids
        elif self.sampling_approach == 'margin_sampling':
            clf = self.query_learner
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
        elif self.sampling_approach == 'power_margin_sampling':
            clf = self.query_learner
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
            return margin_ids

        elif self.sampling_approach == 'least_confident':
            clf = self.query_learner
            clf.fit(self.Xl, self.yl)
            probas = clf.predict_proba(self.Xu)
            probas_max = probas.max(axis=-1)
            least_confident_ids = np.argsort(probas_max)[:self.num_to_label]
            return least_confident_ids

        elif self.sampling_approach == 'entropy':
            clf = self.query_learner
            clf.fit(self.Xl, self.yl)
            probas = clf.predict_proba(self.Xu)
            # get ids where predicted probabilities for two different classes are closest
            entropies = []
            for i in range(len(probas)):
                entropies.append(stats.entropy(probas[i]))
            entropies = np.array(entropies)
            #print("entropies", entropies)
            entropy_ids = np.argsort(entropies)[-self.num_to_label:]
            return entropy_ids

        elif self.sampling_approach == 'qbc':
            clf = RFEns(n_estimators=10)
            clf.fit(self.Xl, self.yl)
            probas = clf.predict_proba(self.Xu)
            consensus_proba = np.mean(probas, axis=0)
            # get for each instance how often each class was predicted
            preds = np.sum(np.round(probas), axis=0)/len(clfs)
            learner_KL_div = np.zeros((probas.shape[0], probas.shape[1]))
            for i in range(probas.shape[0]):
                for j in range(probas.shape[1]):
                    learner_KL_div[i,j] = stats.entropy(probas[i,j], qk=consensus_proba[j])
            KL = learner_KL_div.max(axis=1)
            KL_ids = np.argsort(KL)[-self.num_to_label:]
            return KL_ids

        elif self.sampling_approach == 'weighted_cluster':
            clf = self.query_learner
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
                #TODO hier gabs noch errors in medium setting, to be checked
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

    def init_empty(self):
        """
        This method initializes the active learning pipeline with an empty labeled set.
        """
        self.get_data()
        self.get_split_indices()
        if self.init_query == 'random':
            np.random.seed(self.seed)
            ids = np.random.choice(self.indices_unlabeled, self.num_to_label, replace=False)
        elif self.init_query == 'cluster':
            np.random.seed(self.seed)
            # ids = TODO some clustering bla
        self.indices_labeled = ids
        self.indices_unlabeled = np.setdiff1d(self.indices_train, ids)

    def init(self):
        """
        This method initializes the active learning pipeline.
        """
        self.get_data()
        self.get_split_indices()
        if self.setting == "empty":
            self.init_empty()
        self.get_split_data()
        self.num_classes = len(np.unique(self.yl))
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
            test_auc = None
        test_prec = metrics.precision_score(auc_te, auc_preds, average='macro', zero_division=np.nan)
        test_rec = metrics.recall_score(auc_te, auc_preds, average='macro', zero_division=np.nan)
        self.current_test_metrics = [test_acc, test_auc, test_prec, test_rec]

    def update_indices(self, ids):
        self.indices_labeled = np.concatenate([self.indices_labeled, self.indices_unlabeled[ids]])
        self.indices_unlabeled = np.delete(self.indices_unlabeled, ids)

    def update(self):
        ids = self.get_oracle_ids()
        self.update_indices(ids)
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



def run_active_pipeline():

    import warnings
    from sklearn.exceptions import ConvergenceWarning

    # Suppress FutureWarning for the specified category
    warnings.simplefilter(action='ignore', category=FutureWarning)
    # Filter out ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    data = (openmlid, seed)

    svm_lin = svm(kernel='linear', probability=True)
    svm_rbf = svm(kernel='rbf', probability=True)
    rf_entropy = rf(n_estimators=100, max_depth=10, criterion='entropy')
    rf_gini = rf(n_estimators=100, max_depth=10, criterion='gini')
    knn_3 = nn(n_neighbors=3)
    knn_10 = nn(n_neighbors=10)
    log_reg = lr()
    multinomial_bayes = mnb()
    etc_entropy = etc(n_estimators=100, max_depth=10, criterion='entropy')
    etc_gini = etc(n_estimators=100, max_depth=10, criterion='gini')
    naive_bayes = GaussianNB()
    mlp = MLPClassifier()




    autoAL = ActiveLearningPipeline( )
    autoAL.init()
    init_test_metrics = autoAL.current_test_metrics

    # meta features computen und loggen!
    #autoAL.load_meta_data()
    #meta_feats = autoAL.meta_features
    #print("meta feta shape", meta_feats.shape)

    #total_chars = sum(len(str(s)) for s in meta_feats)


    #print("Total number of characters:", total_chars)    # if result_processor is not None:
    if result_processor is not None:

        result_processor.process_logs({
            'labeling_log': {
                'iteration': 0,
                'num_Xl': autoAL.num_Xl,
                'num_X': autoAL.num_X,
                'Xl_indices': autoAL.indices_labeled,
                'Xu_indices': autoAL.indices_unlabeled,
                'Xtest_indices': autoAL.indices_test
            # 'meta_feats': str(autoAL.meta_features)

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
                    'num_X': autoAL.num_X,
                    'Xl_indices': autoAL.indices_labeled,
                    'Xu_indices': autoAL.indices_unlabeled,
                    'Xtest_indices': autoAL.indices_test
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



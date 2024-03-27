from abc import ABC, abstractmethod
from skactiveml.pool import RandomSampling, UncertaintySampling, EpistemicUncertaintySampling, MonteCarloEER, DiscriminativeAL, QueryByCommittee, BatchBALD
from uncertainty_quantifier import RandomForestEns as RFEns
import numpy as np


class SamplingStrategy(ABC):
    def __init__(self):
        return

    @abstractmethod
    def sample(self, learner, X_l, y_l, X_u, num_samples):
        pass


class WrappedSamplingStrategy(SamplingStrategy):
    def __init__(self, wrapped_strategy: SamplingStrategy, learner):
        self.wrapped_strategy = wrapped_strategy
        self.learner = learner

    def sample(self, learner, X_l, y_l, X_u, num_samples):
        self.learner.fit(X_l, y_l)
        self.wrapped_strategy.sample(self.learner, X_l, y_l, X_u, num_samples)

class PseudoRandomizedSamplingStrategy(SamplingStrategy):
    def __init__(self, seed):
        self.seed = seed

class RandomSamplingStrategy(SamplingStrategy):
    def __init__(self, seed):
        self.qs = RandomSampling(random_state=seed)

    def sample(self, learner, X_l, y_l, X_u, num_samples):
        #to_query_from = np.setdiff1d(np.arange(len(X_u)), already_queried_ids)
        Xu = X_u#[to_query_from]
        nan_labels = np.full(len(Xu), np.nan, dtype=float)
        queried_ids = self.qs.query(X=np.concatenate([X_l, Xu]), y=np.concatenate([y_l, nan_labels]), batch_size=num_samples)
        ids = queried_ids - len(y_l)
        queried_original_ids = to_query_from[ids]
        return queried_original_ids

class EntropySampling(SamplingStrategy):
    def __init__(self, seed, method='entropy'):
        self.qs = UncertaintySampling(method=method, random_state=seed)

    def sample(self, learner, X_l, y_l, X_u, num_samples):
        #to_query_from = np.setdiff1d(np.arange(len(X_u)), already_queried_ids)
        Xu = X_u#[to_query_from]
        nan_labels = np.full(len(Xu), np.nan, dtype=float)
        queried_ids = self.qs.query(X=np.concatenate([X_l, Xu]), y=np.concatenate([y_l, nan_labels]), clf=learner, fit_clf=True, batch_size=num_samples)
        ids = queried_ids - len(y_l)
        queried_original_ids = to_query_from[ids]
        return queried_original_ids


class MarginSampling(EntropySampling):
    def __init__(self, seed):
        super().__init__(seed, method='margin_sampling')

class LeastConfidentSampling(EntropySampling):
    def __init__(self, seed):
        super().__init__(seed, method='least_confident')



class ExpectedAveragePrecision(EntropySampling):
    def __init__(self, seed):
        super().__init__(seed, method='expected_average_precision')

class EpistemicUncertaintySampling(SamplingStrategy):
    def __init__(self, seed):
        super().__init__(seed)
        self.qs = EpistemicUncertaintySampling(random_state=seed)


    def sample(self, learner, X_l, y_l, X_u, num_samples):
        # if learner is not logistic regression or parzen window classifier, raise error
        if learner.__class__.__name__ not in ['LogisticRegression', 'ParzenWindowClassifier']:
            raise ValueError(f"Epistemic uncertainty sampling can only be used with LogisticRegression or ParzenWindowClassifier, not {learner.__class__.__name__}")
        #to_query_from = np.setdiff1d(np.arange(len(X_u)), already_queried_ids)
        Xu = X_u#[to_query_from]
        nan_labels = np.full(len(Xu), np.nan, dtype=float)
        queried_ids = self.qs.query(X=np.concatenate([X_l, Xu]), y=np.concatenate([y_l, nan_labels]), clf=learner, fit_clf=True, batch_size=num_samples)
        ids = queried_ids - len(y_l)
        queried_original_ids = to_query_from[ids]
        return queried_original_ids

class MonteCarloEERLogLoss(SamplingStrategy):
    def __init__(self, seed, method='log_loss'):
        self.qs = MonteCarloEER(method=method, random_state=seed)

    def sample(self, learner, X_l, y_l, X_u, num_samples):
        #to_query_from = np.setdiff1d(np.arange(len(X_u)), already_queried_ids)
        Xu = X_u#[to_query_from]
        nan_labels = np.full(len(Xu), np.nan, dtype=float)
        queried_ids = self.qs.query(X=np.concatenate([X_l, Xu]), y=np.concatenate([y_l, nan_labels]), clf=learner, fit_clf=True, batch_size=num_samples)
        ids = queried_ids - len(y_l)
        queried_original_ids = to_query_from[ids]
        return queried_original_ids

class MonteCarloEERMisclassification(MonteCarloEERLogLoss):
    def __init__(self, seed):
        super().__init__(seed, method='misclassification_loss')


class DiscriminativeSampling(SamplingStrategy):
    def __init__(self, seed):
        self.qs = DiscriminativeAL(random_state=seed)

    def sample(self, learner, X_l, y_l, X_u, num_samples):
        #to_query_from = np.setdiff1d(np.arange(len(X_u)), already_queried_ids)
        Xu = X_u#[to_query_from]
        nan_labels = np.full(len(Xu), np.nan, dtype=float)
        queried_ids = self.qs.query(X=np.concatenate([X_l, Xu]), y=np.concatenate([y_l, nan_labels]), discriminator=learner, batch_size=num_samples)
        ids = queried_ids - len(y_l)
        queried_original_ids = to_query_from[ids]
        return queried_original_ids

class QueryByCommitteeEntropySampling(SamplingStrategy):
    def __init__(self, seed, method='vote_entropy'):
        self.qs = QueryByCommittee(method=method, random_state=seed)

    def sample(self, learner, X_l, y_l, X_u, num_samples):
        learners = [learner]*10
        #to_query_from = np.setdiff1d(np.arange(len(X_u)), already_queried_ids)
        Xu = X_u#[to_query_from]
        nan_labels = np.full(len(Xu), np.nan, dtype=float)
        queried_ids = self.qs.query(X=np.concatenate([X_l, Xu]), y=np.concatenate([y_l, nan_labels]), ensemble=learners, fit_ensemble=True, batch_size=num_samples)
        ids = queried_ids - len(y_l)
        queried_original_ids = to_query_from[ids]
        return queried_original_ids


class QueryByCommitteeKLSampling(QueryByCommitteeEntropySampling):
    def __init__(self, seed):
        super().__init__(seed, method='KL_divergence')

class BatchBaldSampling(SamplingStrategy):
    def __init__(self, seed):
        self.qs = BatchBALD(random_state=seed)

    def sample(self, learner, X_l, y_l, X_u, num_samples):
        learners = [learner]*10
        #to_query_from = np.setdiff1d(np.arange(len(X_u)), already_queried_ids)
        Xu = X_u#[to_query_from]
        nan_labels = np.full(len(Xu), np.nan, dtype=float)
        queried_ids = self.qs.query(X=np.concatenate([X_l, Xu]), y=np.concatenate([y_l, nan_labels]), ensemble=learners, fit_ensemble=True, batch_size=num_samples)
        ids = queried_ids - len(y_l)
        queried_original_ids = to_query_from[ids]
        return queried_original_ids




#########################
# self-implemented
#########################
class TypicalClusterSampling(SamplingStrategy):
    def __init__(self, seed):
        self.seed = seed

    def sample(self, learner, X_l, y_l, X_u, num_samples):
        #to_query_from = np.setdiff1d(np.arange(len(X_u)), already_queried_ids)
        Xu = X_u#[to_query_from]
        # from num_saples "uncovered" cluster (there where are no X_l) select the one with highest "typicality"
        pool_size = len(y_l)
        num_cluster = pool_size + num_samples
        from sklearn.cluster import KMeans
        from sklearn.metrics.pairwise import euclidean_distances
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kmeans = KMeans(n_clusters=num_cluster)
            X = np.concatenate((X_l, Xu))
            kmeans.fit(X)
            class_ids = kmeans.labels_
            labeled_cluster_classes = np.unique(kmeans.labels_[:pool_size])
            cluster_sizes = [len(np.argwhere(kmeans.labels_ == i)) for i in range(num_cluster)]
            ids_by_size = np.argsort(-np.array(cluster_sizes))
            ct = 0
            selected_ids = []
            for idx in ids_by_size:
                if idx not in labeled_cluster_classes:
                    instances_ids = np.argwhere(kmeans.labels_ == idx)
                    instances = X[instances_ids]
                    # compute typicality for each instance, append the one with highest typicality
                    typicalities = []
                    K=20
                    for i,instance in enumerate(instances):
                        remaining_instances = np.delete(instances, i, axis=0)
                        dists = np.sqrt((instance - remaining_instances)**2)
                        if len(dists) < K:
                            dist = np.mean(dists)
                        else:
                            dist = np.mean(np.argsort(dists)[:K])
                        typicality = 1 / dist
                        typicalities.append(typicality)
                    typicalities = np.array(typicalities)
                    selected_id = instances_ids[np.argmax(typicalities)]
                    selected_ids.append(selected_id[0])
                    ct += 1
                    if ct==num_samples:
                        return to_query_from[selected_ids]


class PowerMarginSampling(SamplingStrategy):
    def __init__(self, seed):
        self.seed = seed

    def sample(self, learner, X_l, y_l, X_u, num_samples):
        #to_query_from = np.setdiff1d(np.arange(len(X_u)), already_queried_ids)
        Xu = X_u#[to_query_from]
        probas = learner.predict_proba(Xu)
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
        margin_ids = np.argsort(margins)[-num_samples:]
        original_margin_ids = to_query_from[margin_ids]
        return original_margin_ids

class WeightedClusterSampling(SamplingStrategy):
    def __init__(self, seed):
        self.seed = seed

    def sample(self, learner, X_l, y_l, X_u, num_samples):
        #to_query_from = np.setdiff1d(np.arange(len(X_u)), already_queried_ids)
        Xu = X_u#[to_query_from]
        scores = learner.predict_proba(Xu) + 1e-8
        entropy = -np.sum(scores * np.log(scores), axis=1)
        num_classes = len(np.unique(y_l))
        from sklearn.cluster import KMeans
        from sklearn.metrics.pairwise import euclidean_distances
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kmeans = KMeans(n_clusters=num_classes)
            kmeans.fit(Xu, sample_weight=entropy)
            #TODO hier gabs noch errors in medium setting, to be checked
            class_ids = [np.argwhere(kmeans.labels_ == i) for i in range(num_classes)]
            centers = kmeans.cluster_centers_
            dists = euclidean_distances(centers, Xu)
            sort_idxs = dists.argsort(axis=1)
            q_idxs = []
            n = len(Xu)
            # taken from https://github.com/virajprabhu/CLUE/blob/main/CLUE.py
            ax, rem = 0, n
            idxs_unlabeled = np.arange(n)
            while rem > 0:
                q_idxs.extend(list(sort_idxs[:, ax][:rem]))
                q_idxs = list(set(q_idxs))
                rem = n - len(q_idxs)
                ax += 1
            print("q_idxs: ", q_idxs)
            idxs = idxs_unlabeled[q_idxs[:num_samples]]
            print("idxs: ", idxs)
            return to_query_from[idxs]


class RandomMarginSampling(SamplingStrategy):
    def __init__(self, seed):
        self.seed = seed


    def sample(self, learner, X_l, y_l, X_u, num_samples):
        num_for_margin = num_samples // 2
        num_for_random = num_samples - num_for_margin
        margin_ids = MarginSampling(self.seed).sample(learner, X_l, y_l, X_u, already_queried_ids, num_for_margin)
        random_ids = RandomSamplingStrategy(self.seed).sample(learner, X_l, y_l, X_u, already_queried_ids, num_for_random)
        return np.concatenate((margin_ids, random_ids))

class MinMarginSampling(SamplingStrategy):
    def __init__(self, seed):
        self.seed = seed

    def sample(self, learner, X_l, y_l, X_u, num_samples):
        #to_query_from = np.setdiff1d(np.arange(len(X_u)), already_queried_ids)
        Xu = X_u#[to_query_from]
        num_estimators = 25
        clf = RFEns(n_estimators=num_estimators)
        clf.fit(X_l, y_l)
        all_probas = clf.predict_proba(Xu)
        margins = np.zeros((num_estimators, all_probas.shape[0]))
        for i in range(num_estimators):
            probas = all_probas[:, :, i]
            for j in range(len(probas)):
                sorted = np.sort(probas[i])
                most_likely_prob = sorted[-1]
                second_most_likely_prob = sorted[-2]
                margins[i,j] = most_likely_prob - second_most_likely_prob
        margins = np.min(margins, axis=0)
        margin_ids = np.argsort(margins)[:num_samples]
        return to_query_from[margin_ids]

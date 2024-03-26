from abc import ABC, abstractmethod
import numpy as np
import scipy.stats as stats
from uncertainty_quantifier import RandomForestEns as RFEns

class SamplingStrategy(ABC):
    def __init__(self):
        return

    @abstractmethod
    def sample(self, learner, X_l, y_l, X_u, num_samples):
        pass

class WrappedSamplingStrategy(SamplingStrategy):
    def __init__(self, wrapped_strategy:SamplingStrategy, learner):
        self.wrapped_strategy = wrapped_strategy
        self.learner = learner

    def sample(self, learner, X_l, y_l, X_u, num_samples):
        self.learner.fit(X_l, y_l)
        self.wrapped_strategy.sample(self.learner, X_l, y_l, X_u, num_samples)

class PseudoRandomizedSamplingStrategy(SamplingStrategy):
    def __init__(self, seed):
        self.seed = seed

class RandomSampling(SamplingStrategy):
    def __init__(self, seed):
        super().__init__(seed)

    def sample(self, learner, X_l, y_l, X_u, num_samples):
        np.random.seed(self.seed)
        return np.random.choice(np.arange(len(X_u)), num_samples, replace=False)

class MarginSampling(SamplingStrategy):
    def __init__(self, seed):
        super().__init__(seed)

    def sample(self, learner, X_l, y_l, X_u, num_samples):
        probas = learner.predict_proba(X_u)
        margins = []
        for i in range(len(probas)):
            sorted = np.sort(probas[i])
            most_likely_prob = sorted[-1]
            second_most_likely_prob = sorted[-2]
            margins.append(most_likely_prob - second_most_likely_prob)
        margins = np.array(margins)
        margin_ids = np.argsort(margins)[:num_samples]
        return margin_ids

class RandomMarginSampling(SamplingStrategy):
    def __init__(self):
        super().__init__()

    def sample(self, learner, X_l, y_l, X_u, num_samples):
        num_for_margin = num_samples // 2
        num_for_random = num_samples - num_for_margin
        margin_ids = MarginSampling().sample(learner, X_l, y_l, X_u, num_for_margin)
        random_ids = RandomSampling().sample(learner, X_l, y_l, X_u, num_for_random)
        return np.concatenate((margin_ids, random_ids))

class MinMarginSampling(SamplingStrategy):
    def __init__(self, seed):
        super().__init__(seed)

    def sample(self, learner, X_l, y_l, X_u, num_samples):
        num_estimators = 25
        clf = RFEns(n_estimators=num_estimators)
        clf.fit(X_l, y_l)
        all_probas = clf.predict_proba(X_u)
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
        return margin_ids

class PowerMarginSampling(SamplingStrategy):
    def __init__(self, seed):
        super().__init__(seed)

    def sample(self, learner, X_l, y_l, X_u, num_samples):
        probas = learner.predict_proba(X_u)
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
        return margin_ids

class LeastConfidentSampling(SamplingStrategy):
    def __init__(self, seed):
        super().__init__(seed)

    def sample(self, learner, X_l, y_l, X_u, num_samples):
        probas = learner.predict_proba(X_u)
        probas_max = probas.max(axis=-1)
        least_confident_ids = np.argsort(probas_max)[:num_samples]
        return least_confident_ids


class EntropySampling(SamplingStrategy):
    def __init__(self, seed):
        super().__init__(seed)

    def sample(self, learner, X_l, y_l, X_u, num_samples):
        probas = learner.predict_proba(X_u)
        entropies = []
        for i in range(len(probas)):
            entropies.append(stats.entropy(probas[i]))
        entropies = np.array(entropies)
        # print("entropies", entropies)
        entropy_ids = np.argsort(entropies)[-num_samples:]
        return entropy_ids

class QueryByCommitteeSampling(SamplingStrategy):
    def __init__(self, seed):
        super().__init__(seed)

    def sample(self, learner, X_l, y_l, X_u, num_samples):
        num_estimators = 25
        clf = RFEns(n_estimators=num_estimators)
        clf.fit(X_l, y_l)
        probas = clf.predict_proba(X_u)
        consensus_proba = np.mean(probas, axis=0)
        # get for each instance how often each class was predicted
        preds = np.sum(np.round(probas), axis=0) / len(clfs)
        learner_KL_div = np.zeros((probas.shape[0], probas.shape[1]))
        for i in range(probas.shape[0]):
            for j in range(probas.shape[1]):
                learner_KL_div[i, j] = stats.entropy(probas[i, j], qk=consensus_proba[j])
        KL = learner_KL_div.max(axis=1)
        KL_ids = np.argsort(KL)[-num_samples:]
        return KL_ids

class WeightedClusterSampling(SamplingStrategy):
    def __init__(self, seed):
        super().__init__(seed)

    def sample(self, learner, X_l, y_l, X_u, num_samples):
        scores = clf.predict_proba(X_u) + 1e-8
        entropy = -np.sum(scores * np.log(scores), axis=1)
        num_classes = len(y_l.unique())
        from sklearn.cluster import KMeans
        from sklearn.metrics.pairwise import euclidean_distances
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kmeans = KMeans(n_clusters=num_classes)
            kmeans.fit(X_u, sample_weight=entropy)
            #TODO hier gabs noch errors in medium setting, to be checked
            class_ids = [np.argwhere(kmeans.labels_ == i) for i in range(num_classes)]
            centers = kmeans.cluster_centers_
            dists = euclidean_distances(centers, X_u)
            sort_idxs = dists.argsort(axis=1)
            q_idxs = []
            n = len(X_u)
            # taken from https://github.com/virajprabhu/CLUE/blob/main/CLUE.py
            ax, rem = 0, n
            idxs_unlabeled = np.arange(n)
            while rem > 0:
                q_idxs.extend(list(sort_idxs[:, ax][:rem]))
                q_idxs = list(set(q_idxs))
                rem = n - len(q_idxs)
                ax += 1
            return idxs_unlabeled[q_idxs[:num_samples]]


class TypicalClusterSampling(SamplingStrategy):
    def __init__(self, seed):
        super().__init__(seed)

    def sample(self, learner, X_l, y_l, X_u, num_samples):
        # from num_saples "uncovered" cluster (there where are no X_l) select the one with highest "typicality"
        pool_size = len(y_l)
        num_cluster = pool_size + num_samples
        from sklearn.cluster import KMeans
        from sklearn.metrics.pairwise import euclidean_distances
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kmeans = KMeans(n_clusters=num_cluster)
            X = np.concatenate((X_l, X_u))
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
                    for instance in instances:
                        remaining_instances = np.delete(instances, instance, axis=0)
                        dists = np.linalg.norm(instance - remaining_instances)
                        if len(dists) < K:
                            dist = np.mean(dists)
                        else:
                            dist = np.mean(np.argsort(dists)[:K])
                        typicality = 1 / dist
                        typicalities.append(typicality)
                    typicalities = np.array(typicalities)
                    selected_id = instances_ids[np.argmax(typicalities)]
                    selected_ids.append(selected_id)
                    if ct==num_samples:
                        return selected_ids
                    ct += 1


class CoreSetSampling(SamplingStrategy):
    def __init__(self, seed):
        super().__init__(seed)

    def sample(self, learner, X_l, y_l, X_u, num_samples):


class MutualInformation??

margin density
cluster margin






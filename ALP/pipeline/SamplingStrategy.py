from abc import ABC, abstractmethod

import numpy as np
from skactiveml.pool import (
    BatchBALD,
    DiscriminativeAL,
    EpistemicUncertaintySampling,
    MonteCarloEER,
    QueryByCommittee,
    RandomSampling,
    UncertaintySampling,
)

from uncertainty_quantifier import RandomForestEns as RFEns


class SamplingStrategy(ABC):
    @abstractmethod
    def sample(self, learner, X_l, y_l, X_u, num_samples):
        pass


class PseudoRandomizedSamplingStrategy(SamplingStrategy):
    def __init__(self, seed):
        self.seed = seed

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


class ActiveMLSamplingStrategy(PseudoRandomizedSamplingStrategy):
    def __init__(self, seed, qs):
        super().__init__(seed=seed)
        self.qs = qs

    def sample(self, learner, X_l, y_l, X_u, num_samples):
        nan_labels = np.full(len(X_u), np.nan, dtype=float)
        queried_ids = self.qs.query(X=np.concatenate([X_l, X_u]), y=np.concatenate([y_l, nan_labels]),
                                    batch_size=num_samples)
        queried_original_ids = queried_ids - len(y_l)
        return queried_original_ids


class RandomSamplingStrategy(ActiveMLSamplingStrategy):
    def __init__(self, seed):
        super().__init__(seed, RandomSampling(random_state=seed))


class UncertaintySamplingStrategy(ActiveMLSamplingStrategy):
    def __init__(self, seed, method):
        super().__init__(seed=seed, qs=UncertaintySampling(method=method, random_state=seed))


class EntropySampling(UncertaintySamplingStrategy):
    def __init__(self, seed):
        super().__init__(seed=seed, method='entropy')


class MarginSampling(UncertaintySamplingStrategy):
    def __init__(self, seed):
        super().__init__(seed=seed, method='margin_sampling')


class LeastConfidentSampling(UncertaintySamplingStrategy):
    def __init__(self, seed):
        super().__init__(seed=seed, method='least_confident')


class ExpectedAveragePrecision(UncertaintySamplingStrategy):
    def __init__(self, seed):
        super().__init__(seed=seed, method='expected_average_precision')


class EpistemicUncertaintySamplingStrategy(ActiveMLSamplingStrategy):
    def __init__(self, seed):
        super().__init__(seed, EpistemicUncertaintySampling(random_state=seed))


class MonteCarloEERStrategy(ActiveMLSamplingStrategy):
    def __init__(self, seed, method):
        super().__init__(seed, MonteCarloEER(method=method, random_state=seed))


class MonteCarloEERLogLoss(MonteCarloEERStrategy):
    def __init__(self, seed):
        super().__init__(seed, method='log_loss')


class MonteCarloEERMisclassification(MonteCarloEERStrategy):
    def __init__(self, seed):
        super().__init__(seed, method='misclassification_loss')


class DiscriminativeSampling(ActiveMLSamplingStrategy):
    def __init__(self, seed):
        super().__init__(seed=seed, qs=DiscriminativeAL(random_state=seed))


class ActiveMLEnsembleSamplingStrategy(ActiveMLSamplingStrategy):
    def __init__(self, seed, qs, ensemble_size):
        super().__init__(seed=seed, qs=qs)
        self.ensemble_size = ensemble_size

    def sample(self, learner, X_l, y_l, X_u, num_samples):
        learners = [learner] * self.ensemble_size
        nan_labels = np.full(len(X_u), np.nan, dtype=float)
        queried_ids = self.qs.query(X=np.concatenate([X_l, X_u]), y=np.concatenate([y_l, nan_labels]),
                                    ensemble=learners, fit_ensemble=True, batch_size=num_samples)
        queried_original_ids = queried_ids - len(y_l)
        return queried_original_ids


class QueryByCommitteeSampling(ActiveMLEnsembleSamplingStrategy):
    def __init__(self, seed, method, ensemble_size):
        super().__init__(seed=seed, qs=QueryByCommittee(method=method, random_state=seed), ensemble_size=ensemble_size)


class QueryByCommitteeEntropySampling(QueryByCommitteeSampling):
    def __init__(self, seed, ensemble_size):
        super().__init__(seed=seed, method='vote_entropy', ensemble_size=ensemble_size)


class QueryByCommitteeKLSampling(QueryByCommitteeSampling):
    def __init__(self, seed, ensemble_size):
        super().__init__(seed=seed, method='KL_divergence', ensemble_size=ensemble_size)


class BatchBaldSampling(ActiveMLEnsembleSamplingStrategy):
    def __init__(self, seed, ensemble_size):
        super().__init__(seed=seed, qs=BatchBALD(random_state=seed), ensemble_size=ensemble_size)


#########################
# self-implemented
#########################
class TypicalClusterSampling(SamplingStrategy):
    def __init__(self, seed):
        self.seed = seed

    def sample(self, learner, X_l, y_l, X_u, num_samples):
        # from num_saples "uncovered" cluster (there where are no X_l) select the one with highest "typicality"
        pool_size = len(y_l)
        num_cluster = pool_size + num_samples
        import warnings

        from sklearn.cluster import KMeans
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kmeans = KMeans(n_clusters=num_cluster)
            X = np.concatenate((X_l, X_u))
            kmeans.fit(X)
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
                    K = 20
                    for i, instance in enumerate(instances):
                        remaining_instances = np.delete(instances, i, axis=0)
                        dists = np.sqrt((instance - remaining_instances) ** 2)
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
                    if ct == num_samples:
                        return selected_ids


class PowerMarginSampling(SamplingStrategy):
    def __init__(self, seed):
        self.seed = seed

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
        original_margin_ids = np.argsort(margins)[-num_samples:]
        return original_margin_ids


class WeightedClusterSampling(SamplingStrategy):
    def __init__(self, seed):
        self.seed = seed

    def sample(self, learner, X_l, y_l, X_u, num_samples):
        print("X_u: ", X_u.shape)
        scores = learner.predict_proba(X_u) + 1e-8
        entropy = -np.sum(scores * np.log(scores), axis=1)
        num_classes = len(np.unique(y_l))
        import warnings

        from sklearn.cluster import KMeans
        from sklearn.metrics.pairwise import euclidean_distances
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kmeans = KMeans(n_clusters=num_classes)
            kmeans.fit(X_u, sample_weight=entropy)
            centers = kmeans.cluster_centers_
            # for each class, find instance clostes to center
            selected_ids = []

            per_class = num_samples // num_classes + 1
            for class_ in range(num_classes):
                class_ids = np.argwhere(kmeans.labels_ == class_)
                # corresponding center
                center = centers[class_].reshape(1, -1)
                class_neigh = np.squeeze(X_u[class_ids], axis=1)
                dists = euclidean_distances(center, class_neigh)
                closest_instances_id = np.argsort(dists)[:per_class][0]
                for closest_instance_id in closest_instances_id:
                    selected_ids.append(class_ids[closest_instance_id][0])

            return selected_ids[0:num_samples]


class RandomMarginSampling(SamplingStrategy):
    def __init__(self, seed):
        self.seed = seed

    def sample(self, learner, X_l, y_l, X_u, num_samples):
        num_for_margin = num_samples // 2
        num_for_random = num_samples - num_for_margin
        margin_ids = MarginSampling(self.seed).sample(learner, X_l, y_l, X_u, num_for_margin)
        random_ids = RandomSamplingStrategy(self.seed).sample(learner, X_l, y_l, X_u, num_for_random)
        return np.concatenate((margin_ids, random_ids))


class MinMarginSampling(SamplingStrategy):
    def __init__(self, seed):
        self.seed = seed

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
                margins[i, j] = most_likely_prob - second_most_likely_prob
        margins = np.min(margins, axis=0)
        margin_ids = np.argsort(margins)[:num_samples]
        return margin_ids

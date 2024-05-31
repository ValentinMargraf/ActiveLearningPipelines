import time
import warnings
from abc import ABC, abstractmethod

import numpy as np
from skactiveml.classifier import SklearnClassifier
from skactiveml.pool import (
    BatchBALD,
    DiscriminativeAL,
    EpistemicUncertaintySampling,
    MonteCarloEER,
    QueryByCommittee,
    RandomSampling,
    UncertaintySampling,
)
from sklearn.cluster import AgglomerativeClustering, KMeans

from ALP.util.common import fullname
from ALP.util.ensemble_constructor import Ensemble as Ens


class SamplingStrategy(ABC):
    @abstractmethod
    def sample(self, learner, X_l, y_l, X_u, num_samples):
        pass

    @abstractmethod
    def get_params(self):
        pass


class PseudoRandomizedSamplingStrategy(SamplingStrategy):
    def __init__(self, seed):
        self.seed = seed

    @abstractmethod
    def sample(self, learner, X_l, y_l, X_u, num_samples):
        pass

    def get_params(self):
        return {"seed": self.seed}


class WrappedSamplingStrategy(SamplingStrategy):
    def __init__(self, wrapped_strategy: SamplingStrategy, learner):
        self.wrapped_strategy = wrapped_strategy
        self.learner = learner

    def sample(self, learner, X_l, y_l, X_u, num_samples):
        self.learner.fit(X_l, y_l)
        self.wrapped_strategy.sample(self.learner, X_l, y_l, X_u, num_samples)

    def get_params(self):
        return {
            "wrapped_sampling_strategy": {
                "fqn": fullname(self.wrapped_strategy),
                "params": self.wrapped_strategy.get_params(),
            },
            "learner": {"fqn": fullname(self.learner), "params": self.learner.get_params()},
        }


class ActiveMLSamplingStrategy(PseudoRandomizedSamplingStrategy):
    def __init__(self, seed, qs):
        super().__init__(seed=seed)
        self.qs = qs

    def sample(self, learner, X_l, y_l, X_u, num_samples):
        nan_labels = np.full(len(X_u), np.nan, dtype=float)
        queried_ids = self.qs.query(
            X=np.concatenate([X_l, X_u]), y=np.concatenate([y_l, nan_labels]), batch_size=num_samples
        )
        queried_original_ids = queried_ids - len(y_l)

        return queried_original_ids


class ActiveMLModelBasedSamplingStrategy(ActiveMLSamplingStrategy):
    def __init__(self, seed, qs):
        super().__init__(seed=seed, qs=qs)

    def sample(self, learner, X_l, y_l, X_u, num_samples):
        nan_labels = np.full(len(X_u), np.nan, dtype=float)
        queried_ids = self.qs.query(
            clf=SklearnClassifier(learner),
            X=np.concatenate([X_l, X_u]),
            y=np.concatenate([y_l, nan_labels]),
            batch_size=num_samples,
        )
        queried_original_ids = queried_ids - len(y_l)

        return queried_original_ids


class RandomSamplingStrategy(ActiveMLSamplingStrategy):
    def __init__(self, seed):
        super().__init__(seed, RandomSampling(random_state=seed))


class UncertaintySamplingStrategy(ActiveMLModelBasedSamplingStrategy):
    def __init__(self, seed, method):
        super().__init__(seed=seed, qs=UncertaintySampling(method=method, random_state=seed))


class ExpectedAveragePrecision(UncertaintySamplingStrategy):
    def __init__(self, seed):
        super().__init__(seed=seed, method="expected_average_precision")


class EpistemicUncertaintySamplingStrategy(ActiveMLSamplingStrategy):
    def __init__(self, seed):
        super().__init__(seed, EpistemicUncertaintySampling(random_state=seed))


class MonteCarloEERStrategy(ActiveMLModelBasedSamplingStrategy):
    def __init__(self, seed, method):
        super().__init__(seed, MonteCarloEER(method=method, random_state=seed))


class MonteCarloEERLogLoss(MonteCarloEERStrategy):
    def __init__(self, seed):
        super().__init__(seed, method="log_loss")


class MonteCarloEERMisclassification(MonteCarloEERStrategy):
    def __init__(self, seed):
        super().__init__(seed, method="misclassification_loss")


class DiscriminativeSampling(ActiveMLSamplingStrategy):
    def __init__(self, seed):
        super().__init__(seed=seed, qs=DiscriminativeAL(random_state=seed))

    def sample(self, learner, X_l, y_l, X_u, num_samples):
        nan_labels = np.full(len(X_u), np.nan, dtype=float)
        queried_ids = self.qs.query(
            X=np.concatenate([X_l, X_u]),
            y=np.concatenate([y_l, nan_labels]),
            discriminator=SklearnClassifier(learner),
            batch_size=num_samples,
        )
        queried_original_ids = queried_ids - len(y_l)
        return queried_original_ids


class ActiveMLEnsembleSamplingStrategy(ActiveMLSamplingStrategy):
    def __init__(self, seed, qs, ensemble_size):
        super().__init__(seed=seed, qs=qs)
        self.ensemble_size = ensemble_size

    def sample(self, learner, X_l, y_l, X_u, num_samples):
        learners = [SklearnClassifier(learner)] * self.ensemble_size
        nan_labels = np.full(len(X_u), np.nan, dtype=float)
        queried_ids = self.qs.query(
            X=np.concatenate([X_l, X_u]),
            y=np.concatenate([y_l, nan_labels]),
            ensemble=learners,
            fit_ensemble=True,
            batch_size=num_samples,
        )
        queried_original_ids = queried_ids - len(y_l)
        return queried_original_ids

    def get_params(self):
        params = super().get_params()
        params["ensemble_size"] = self.ensemble_size
        return params


class QueryByCommitteeSampling(ActiveMLEnsembleSamplingStrategy):
    def __init__(self, seed, method, ensemble_size):
        super().__init__(seed=seed, qs=QueryByCommittee(method=method, random_state=seed), ensemble_size=ensemble_size)


class EnsemblePseudoRandomizedSampling(PseudoRandomizedSamplingStrategy):
    def __init__(self, seed, ensemble_size):
        super().__init__(seed=seed)
        self.ensemble_size = ensemble_size

    @abstractmethod
    def sample(self, learner, X_l, y_l, X_u, num_samples):
        pass

    def get_params(self):
        return {"seed": self.seed, "ensemble_size": self.ensemble_size}


class QueryByCommitteeEntropySampling(QueryByCommitteeSampling):
    def __init__(self, seed, ensemble_size):
        super().__init__(seed=seed, method="vote_entropy", ensemble_size=ensemble_size)


class QueryByCommitteeKLSampling(QueryByCommitteeSampling):
    def __init__(self, seed, ensemble_size):
        super().__init__(seed=seed, method="KL_divergence", ensemble_size=ensemble_size)


class BatchBaldSampling(ActiveMLEnsembleSamplingStrategy):
    def __init__(self, seed, ensemble_size):
        super().__init__(seed=seed, qs=BatchBALD(random_state=seed), ensemble_size=ensemble_size)


#########################
# self-implemented
#########################
class EmbeddingBasedSampling(PseudoRandomizedSamplingStrategy):
    def __init__(self, seed):
        super().__init__(seed=seed)

    def compute_embedding(self, learner, X_l, y_l, X_u, transform_labeled=False):
        learner_fqn = fullname(learner)
        if learner_fqn == "tabpfn.scripts.transformer_prediction_interface.TabPFNClassifier":
            from ALP.util.TorchUtil import TabPFNEmbedder
            clf = TabPFNEmbedder(X_l, y_l)
            if not transform_labeled:
                X_u = clf.forward(X_u, encode=True)
            else:
                X = np.concatenate((X_l, X_u))
                X_embeds = clf.forward(X, encode=True)
                X_l, X_u = X_embeds[: len(X_l)], X_embeds[len(X_l):]
        elif learner_fqn == "pytorch_tabnet.tab_model.TabNetClassifier":
            from pytorch_tabnet.tab_model import TabNetClassifier
            clf = TabNetClassifier(verbose=0)
            from ALP.util.TorchUtil import TimeLimitCallback
            clf.fit(X_l, y_l, callbacks=[TimeLimitCallback(180)])
            if not transform_labeled:
                X_u = clf.predict_proba(X_u, get_embeds=True)
            else:
                X = np.concatenate((X_l, X_u))
                X_embeds = clf.predict_proba(X, get_embeds=True)
                X_l, X_u = X_embeds[: len(X_l)], X_embeds[len(X_l):]
        if not transform_labeled:
            return X_u
        else:
            return X_u, X_l


class TypicalClusterSampling(EmbeddingBasedSampling):
    def __init__(self, seed):
        super().__init__(seed=seed)

    def sample(self, learner, X_l, y_l, X_u, num_samples):
        # from num_saples "uncovered" cluster (there where are no X_l) select the one with highest "typicality"
        pool_size = len(y_l)
        num_cluster = pool_size + num_samples
        X_u, X_l = self.compute_embedding(learner, X_l, y_l, X_u, transform_labeled=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kmeans = KMeans(n_clusters=num_cluster)
            X = np.concatenate((X_l, X_u))
            kmeans.fit(X)
            labeled_cluster_classes = np.unique(kmeans.labels_[:pool_size])
            cluster_sizes = [len(np.argwhere(kmeans.labels_ == i)) for i in range(num_cluster)]
            ids_by_size = np.argsort(-np.array(cluster_sizes))
            label_of_ids_by_size = np.arange(num_cluster)[ids_by_size]
            ct = 0
            selected_ids = []
            for idx in ids_by_size:
                if label_of_ids_by_size[idx] not in labeled_cluster_classes:
                    instances_ids = np.argwhere(kmeans.labels_ == label_of_ids_by_size[idx])
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
                    if len(typicalities) > 0:
                        selected_id = instances_ids[np.argmax(typicalities)]
                        selected_ids.append(selected_id[0])
                        ct += 1
                        if ct == num_samples:
                            return np.array(selected_ids) - pool_size


class LeastConfidentSampling(PseudoRandomizedSamplingStrategy):
    def __init__(self, seed):
        super().__init__(seed=seed)

    def sample(self, learner, X_l, y_l, X_u, num_samples):
        probas = learner.predict_proba(X_u)
        least_confidence = 1 - np.max(probas, axis=1)
        least_confidence_ids = np.argsort(least_confidence)[-num_samples:]
        return least_confidence_ids


class EntropySampling(PseudoRandomizedSamplingStrategy):
    def __init__(self, seed):
        super().__init__(seed=seed)

    def sample(self, learner, X_l, y_l, X_u, num_samples):
        probas = learner.predict_proba(X_u)
        entropies = -np.sum(probas * np.log(probas + 1e-8), axis=1)
        entropy_ids = np.argsort(entropies)[-num_samples:]
        return entropy_ids


class MarginSampling(PseudoRandomizedSamplingStrategy):
    def __init__(self, seed):
        super().__init__(seed=seed)

    def sample(self, learner, X_l, y_l, X_u, num_samples):
        probas = learner.predict_proba(X_u)
        sorted_probas = np.sort(probas, axis=-1)
        margins = sorted_probas[:, -1] - sorted_probas[:, -2]
        original_margin_ids = np.argsort(margins)[:num_samples]
        return original_margin_ids


class PowerMarginSampling(PseudoRandomizedSamplingStrategy):
    def __init__(self, seed):
        super().__init__(seed=seed)

    def sample(self, learner, X_l, y_l, X_u, num_samples):
        probas = learner.predict_proba(X_u)
        sorted_probas = np.sort(probas, axis=-1)
        margins = sorted_probas[:, -1] - sorted_probas[:, -2]
        margins = 1 - np.array(margins)
        # power transform
        np.random.seed(self.seed)
        margins = np.log(margins + 1e-8) + np.random.gumbel(size=len(margins))
        original_margin_ids = np.argsort(margins)[-num_samples:]
        return original_margin_ids


class RandomMarginSampling(PseudoRandomizedSamplingStrategy):
    def __init__(self, seed):
        super().__init__(seed=seed)

    def sample(self, learner, X_l, y_l, X_u, num_samples):
        num_for_margin = num_samples // 2
        num_for_random = num_samples - num_for_margin
        random_ids = RandomSamplingStrategy(self.seed).sample(learner, X_l, y_l, X_u, num_for_random)
        if num_for_margin == 0:
            return random_ids
        else:
            margin_ids = MarginSampling(self.seed).sample(learner, X_l, y_l, X_u, num_for_margin)
            return np.concatenate((margin_ids, random_ids))


class MinMarginSampling(EnsemblePseudoRandomizedSampling):
    def __init__(self, seed, ensemble_size):
        super().__init__(seed=seed, ensemble_size=ensemble_size)
        self.ensemble_size = ensemble_size

    def sample(self, learner, X_l, y_l, X_u, num_samples):
        clf = Ens(estimator=learner, num_estimators=self.ensemble_size, max_neighbors=len(y_l))
        clf.fit(X_l, y_l)
        all_probas = clf.predict_proba(X_u)
        margins = np.zeros((self.ensemble_size, all_probas.shape[0]))
        sorted_probas = np.sort(all_probas, axis=1)
        margins = sorted_probas[:, -1, :] - sorted_probas[:, -2, :]
        margins = np.min(margins, axis=-1)
        margin_ids = np.argsort(margins)[:num_samples]
        return margin_ids


class FalcunSampling(PseudoRandomizedSamplingStrategy):
    def __init__(self, seed):
        super().__init__(seed=seed)

    def sample(self, learner, X_l, y_l, X_u, num_samples):
        probas = learner.predict_proba(X_u)
        sorted_probas = np.sort(probas, axis=-1)
        margins = sorted_probas[:, -1] - sorted_probas[:, -2]
        div_scores = margins
        gamma = 10
        selected_ids = []
        ids_to_choose_from = np.arange(len(margins))
        mask = np.ones(len(margins), dtype=bool)  # Initialize mask
        for round in range(num_samples):
            relevance = margins + div_scores
            relevance[np.isnan(relevance)] = 0

            np.random.seed(self.seed)
            prob = relevance[mask] ** gamma / np.sum(relevance[mask] ** gamma)
            # set nan to 0, afterwards normalize
            prob[np.isnan(prob)] = 0
            prob = prob / np.sum(prob)
            prob[np.isnan(prob)] = 0

            # if still a nan
            if np.sum(np.isnan(prob)) > 0:
                selected_id = np.random.choice(ids_to_choose_from[mask])
            else:
                selected_id = np.random.choice(ids_to_choose_from[mask], p=prob)
            selected_ids.append(selected_id)
            # print("selected id", selected_id)
            # update div scores
            x_q = probas[selected_id]
            mask[selected_id] = False
            # remove selected id
            distances = np.sum(abs(probas - x_q), axis=1)
            div_scores = np.minimum(div_scores, distances)
            # normalize
            div_scores = (div_scores - np.min(div_scores)) / (np.max(div_scores) - np.min(div_scores) + 1e-8)

        return selected_ids


class WeightedClusterSampling(EmbeddingBasedSampling):
    def __init__(self, seed):
        super().__init__(seed=seed)

    def sample(self, learner, X_l, y_l, X_u, num_samples):
        scores = learner.predict_proba(X_u) + 1e-8
        entropy = -np.sum(scores * np.log(scores), axis=1)
        num_classes = num_samples
        X_u = self.compute_embedding(learner, X_l, y_l, X_u)

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
                # get nearest neighbor
                class_neigh = np.squeeze(X_u[class_ids], axis=1)
                dists = euclidean_distances(center, class_neigh)
                if len(dists) < per_class:
                    closest_instances_id = np.argsort(dists)[:][0]
                else:
                    closest_instances_id = np.argsort(dists)[:per_class][0]
                for closest_instance_id in closest_instances_id:
                    selected_ids.append(class_ids[closest_instance_id][0])

            return selected_ids[0:num_samples]


class KMeansSampling(EmbeddingBasedSampling):
    def __init__(self, seed):
        super().__init__(seed=seed)

    def sample(self, learner, X_l, y_l, X_u, num_samples):
        X_u = self.compute_embedding(learner, X_l, y_l, X_u)
        num_classes = len(np.unique(y_l))
        from sklearn.cluster import KMeans
        from sklearn.metrics.pairwise import euclidean_distances

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kmeans = KMeans(n_clusters=num_classes)

            kmeans.fit(X_u)
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
                if len(dists) < per_class:
                    closest_instances_id = np.argsort(dists)[:][0]
                else:
                    closest_instances_id = np.argsort(dists)[:per_class][0]
                for closest_instance_id in closest_instances_id:
                    selected_ids.append(class_ids[closest_instance_id][0])

            return selected_ids[0:num_samples]


class ClusterMargin(EmbeddingBasedSampling):
    def __init__(self, seed):
        super().__init__(seed=seed)

    def sample(self, learner, X_l, y_l, X_u, num_samples):
        m = 10
        probas = learner.predict_proba(X_u)
        X_u = self.compute_embedding(learner, X_l, y_l, X_u)
        num_clusters = min((len(X_l) + len(X_u)) // m, len(X_u))
        clustering = AgglomerativeClustering(n_clusters=num_clusters).fit(X_u)

        sorted_probas = np.sort(probas, axis=-1)
        margins = sorted_probas[:, -1] - sorted_probas[:, -2]

        # retrieve m*num_samples instances with smallest margins
        num_to_retrieve = m * num_samples
        ids_to_retrieve = np.argsort(margins)[:num_to_retrieve]
        # from those sample diversity-based through clustering
        cluster_belongings = clustering.labels_[ids_to_retrieve]
        # sort cluster by size
        cluster_dict = {}
        # get cluster sizes
        for i in set(cluster_belongings):
            cluster_dict[i] = len(np.argwhere(cluster_belongings == i))
        # sort by size
        keys = np.array(list(cluster_dict.keys()))
        values = np.array(list(cluster_dict.values()))
        sorted_indices = np.argsort(values)
        sorted_keys = keys[sorted_indices]

        selected_ids = []
        upperbound = len(cluster_belongings)
        ct = 0
        while len(selected_ids) < num_samples:
            for key in sorted_keys:
                to_sample_from = np.argwhere(cluster_belongings == key)
                if len(to_sample_from) == 0:
                    ct += 1
                    if len(selected_ids) == num_samples:
                        return selected_ids
                    if ct == upperbound:
                        return selected_ids
                    continue
                else:
                    np.random.seed(self.seed)
                    id = np.random.choice(to_sample_from[0])
                    selected_ids.append(ids_to_retrieve[id])
                    cluster_belongings = np.delete(cluster_belongings, id)
                    ids_to_retrieve = np.delete(ids_to_retrieve, id)
                    ct += 1
                    if len(selected_ids) == num_samples:
                        return selected_ids

        return selected_ids[:num_samples]


class MaxEntropySampling(EnsemblePseudoRandomizedSampling):
    def __init__(self, seed, ensemble_size):
        super().__init__(seed=seed, ensemble_size=25)
        self.ensemble_size = ensemble_size

    def sample(self, learner, X_l, y_l, X_u, num_samples):
        clf = Ens(estimator=learner, num_estimators=self.ensemble_size, max_neighbors=len(y_l))
        clf.fit(X_l, y_l)
        probas = clf.predict_proba(X_u).mean(axis=-1)
        entropies = -np.sum(probas * np.log(probas + 1e-8), axis=1)
        entropy_ids = np.argsort(entropies)[-num_samples:]
        return entropy_ids


class CoreSetSampling(EmbeddingBasedSampling):
    def __init__(self, seed):
        super().__init__(seed=seed)

    def sample(self, learner, X_l, y_l, X_u, num_samples):
        X_u, X_l = self.compute_embedding(learner, X_l, y_l, X_u, transform_labeled=True)
        selected_ids = []
        for round in range(num_samples):
            active_set = X_l[:, np.newaxis, :]
            inactive_set = X_u[np.newaxis, :, :]
            distances = np.linalg.norm(active_set - inactive_set, axis=2)
            # compute distance to closest neighbor
            dists = distances.min(axis=0)
            # get the id of the instance with the highest distance
            selected_id = np.argmax(dists)
            selected_ids.append(selected_id)
            # remove selected id
            X_l = np.concatenate((X_l, X_u[selected_id].reshape(1, -1)))
            X_u = np.delete(X_u, selected_id, axis=0)
        return selected_ids


class BALDSampling(EnsemblePseudoRandomizedSampling):
    def __init__(self, seed, ensemble_size):
        super().__init__(seed=seed, ensemble_size=25)
        self.ensemble_size = ensemble_size

    def sample(self, learner, X_l, y_l, X_u, num_samples):
        clf = Ens(estimator=learner, num_estimators=self.ensemble_size, max_neighbors=len(y_l))
        clf.fit(X_l, y_l)
        probas = clf.predict_proba(X_u)
        probas_mean = probas.mean(axis=-1)
        entropies = -np.sum(probas_mean * np.log(probas_mean + 1e-8), axis=1)
        mean_entropy = np.mean(
            np.array(
                [-np.sum(probas[:, :, i] * np.log(probas[:, :, i] + 1e-8), axis=1) for i in range(self.ensemble_size)]
            ),
            axis=0,
        )
        mutual_info = entropies - mean_entropy
        mutual_info_ids = np.argsort(mutual_info)[-num_samples:]
        return mutual_info_ids


class PowerBALDSampling(EnsemblePseudoRandomizedSampling):
    def __init__(self, seed, ensemble_size):
        super().__init__(seed=seed, ensemble_size=25)
        self.ensemble_size = ensemble_size

    def sample(self, learner, X_l, y_l, X_u, num_samples):
        clf = Ens(estimator=learner, num_estimators=self.ensemble_size, max_neighbors=len(y_l))
        clf.fit(X_l, y_l)
        probas = clf.predict_proba(X_u)
        probas_mean = probas.mean(axis=-1)
        entropies = -np.sum(probas_mean * np.log(probas_mean + 1e-8), axis=1)
        mean_entropy = np.mean(
            np.array(
                [-np.sum(probas[:, :, i] * np.log(probas[:, :, i] + 1e-8), axis=1) for i in range(self.ensemble_size)]
            ),
            axis=0,
        )
        mutual_info = entropies - mean_entropy + np.random.gumbel(size=len(entropies))
        mutual_info_ids = np.argsort(mutual_info)[-num_samples:]
        return mutual_info_ids


class QBCVarianceRatioSampling(EnsemblePseudoRandomizedSampling):
    def __init__(self, seed, ensemble_size):
        super().__init__(seed=seed, ensemble_size=25)
        self.ensemble_size = ensemble_size

    def sample(self, learner, X_l, y_l, X_u, num_samples):
        clf = Ens(estimator=learner, num_estimators=self.ensemble_size, max_neighbors=len(y_l))
        clf.fit(X_l, y_l)
        probas = clf.predict_proba(X_u)
        preds = probas.argmax(axis=1)
        variance_ratios = []
        for i in range(len(preds)):
            unique_labels, counts = np.unique(preds[i], return_counts=True)
            # Find the index of the label that occurs most often
            most_common_index = np.argmax(counts)
            # Get the most common label and its count
            most_common_count = counts[most_common_index]
            variance_ratio = 1 - most_common_count / self.ensemble_size
            variance_ratios.append(variance_ratio)
        variance_ratios = np.array(variance_ratios)
        vr_ids = np.argsort(variance_ratios)[-num_samples:]
        return vr_ids



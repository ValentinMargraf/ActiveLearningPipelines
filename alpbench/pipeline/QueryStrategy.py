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

from alpbench.util.common import fullname
from alpbench.util.ensemble_constructor import Ensemble as Ens


class QueryStrategy(ABC):
    """QueryStrategy

    This class is an abstract class for query strategies. The query strategies are used to sample instances from the
    pool of unlabeled instances.
    """

    @abstractmethod
    def sample(self, learner, X_l, y_l, X_u, num_queries):
        """
        This method samples instances from the pool of unlabeled instances. It is given a learner, that
        is already fitted on the labeled data and potentially used to predict probabilities for the unlabeled
        data.

        Parameters:
            learner: object
            X_l: np.ndarray
            y_l: np.ndarray
            X_u: np.ndarray
            num_queries: int
        """
        pass

    @abstractmethod
    def get_params(self):
        """
        This method returns the parameters of the query strategy.
        """
        pass


class PseudoRandomizedQueryStrategy(QueryStrategy):
    """PseudoRandomizedQueryStrategy

    This class is an abstract class for query strategies that are pseudo-randomized, meaning that they can be
    reproduced with the same random seed.

    Args:
        seed (int): The seed for the random number generator.

    Attributes:
        seed (int): The seed for the random number generator.
    """

    def __init__(self, seed):
        self.seed = seed

    @abstractmethod
    def sample(self, learner, X_l, y_l, X_u, num_queries):
        pass

    def get_params(self):
        return {"seed": self.seed}


class WrappedQueryStrategy(QueryStrategy):
    """WrappedQueryStrategy

    This class is used to wrap a query strategy with a learner. The wrapped query strategy is used to sample instances
    from the pool of unlabeled instances.

    Args:
        wrapped_query_strategy: object
        learner: object

    Attributes:
        wrapped_query_strategy: object
        learner: object
    """

    def __init__(self, wrapped_query_strategy: QueryStrategy, learner):
        self.wrapped_query_strategy = wrapped_query_strategy
        self.learner = learner

    def sample(self, learner, X_l, y_l, X_u, num_queries):
        self.learner.fit(X_l, y_l)
        self.wrapped_query_strategy.sample(self.learner, X_l, y_l, X_u, num_queries)

    def get_params(self):
        return {
            "wrapped_query_strategy": {
                "fqn": fullname(self.wrapped_query_strategy),
                "params": self.wrapped_query_strategy.get_params(),
            },
            "learner": {"fqn": fullname(self.learner), "params": self.learner.get_params()},
        }


class ActiveMLQueryStrategy(PseudoRandomizedQueryStrategy):
    """ActiveMLQueryStrategy

    This class is an abstract class for active learning query strategies. The query strategies are used to sample
    instances from the pool of unlabeled instances.

    Args:
        seed (int): The seed for the random number generator.
        qs: object

    Attributes:
        seed (int): The seed for the random number generator.
        qs: object
    """

    def __init__(self, seed, qs):
        super().__init__(seed=seed)
        self.qs = qs

    def sample(self, learner, X_l, y_l, X_u, num_queries):
        nan_labels = np.full(len(X_u), np.nan, dtype=float)
        queried_ids = self.qs.query(
            X=np.concatenate([X_l, X_u]), y=np.concatenate([y_l, nan_labels]), batch_size=num_queries
        )
        queried_original_ids = queried_ids - len(y_l)
        return queried_original_ids


class ActiveMLModelBasedQueryStrategy(ActiveMLQueryStrategy):
    """ActiveMLModelBasedQueryStrategy

    This class is an abstract class for active learning query strategies that are model-based. The query strategies are
    used to sample instances from the pool of unlabeled instances.

    Args:
        seed (int): The seed for the random number generator.
        qs: object

    Attributes:
        seed (int): The seed for the random number generator.
        qs: object
    """

    def __init__(self, seed, qs):
        super().__init__(seed=seed, qs=qs)

    def sample(self, learner, X_l, y_l, X_u, num_queries):
        nan_labels = np.full(len(X_u), np.nan, dtype=float)
        queried_ids = self.qs.query(
            clf=SklearnClassifier(learner),
            X=np.concatenate([X_l, X_u]),
            y=np.concatenate([y_l, nan_labels]),
            batch_size=num_queries,
        )
        queried_original_ids = queried_ids - len(y_l)

        return queried_original_ids


class RandomQueryStrategy(ActiveMLQueryStrategy):
    """RandomQueryStrategy

    This class is used to sample instances from the pool of unlabeled instances randomly.

    Args:
        seed (int): The seed for the random number generator.

    Attributes:
        seed (int): The seed for the random number generator.
    """

    def __init__(self, seed):
        super().__init__(seed, RandomSampling(random_state=seed))


class UncertaintyQueryStrategy(ActiveMLQueryStrategy):
    """UncertaintyQueryStrategy

    This class is used to sample instances from the pool of unlabeled instances based on uncertainty.

    Args:
        seed (int): The seed for the random number generator.
        method (str): The method for uncertainty sampling.

    Attributes:
        seed (int): The seed for the random number generator.
        qs: object
    """

    def __init__(self, seed, method):
        super().__init__(seed=seed, qs=UncertaintySampling(method=method, random_state=seed))


class ExpectedAveragePrecision(UncertaintyQueryStrategy):
    """ExpectedAveragePrecision

    This class is used to sample instances from the pool of unlabeled instances based on the expected average precision.

    Args:
        seed (int): The seed for the random number generator.

    Attributes:
        seed (int): The seed for the random number generator.
    """

    def __init__(self, seed):
        super().__init__(seed=seed, method="expected_average_precision")


class EpistemicUncertaintyQueryStrategy(ActiveMLModelBasedQueryStrategy):
    """EpistemicUncertaintyQueryStrategy

    This class is used to sample instances from the pool of unlabeled instances based on epistemic uncertainty.

    Args:
        seed (int): The seed for the random number generator.

    Attributes:
        seed (int): The seed for the random number generator.
    """

    def __init__(self, seed):
        super().__init__(seed, EpistemicUncertaintySampling(random_state=seed))


class MonteCarloEERStrategy(ActiveMLModelBasedQueryStrategy):
    """MonteCarloEERStrategy

    This class is used to sample instances from the pool of unlabeled instances based on the Monte Carlo EER method.

    Args:
        seed (int): The seed for the random number generator.
        method (str): The method for Monte Carlo EER.

    Attributes:
        seed (int): The seed for the random number generator.
        qs: object
    """

    def __init__(self, seed, method):
        super().__init__(seed, MonteCarloEER(method=method, random_state=seed))


class MonteCarloEERLogLoss(MonteCarloEERStrategy):
    """MonteCarloEERLogLoss

    This class is used to sample instances from the pool of unlabeled instances based on the Monte Carlo EER method with
    the log loss method.

    Args:
        seed (int): The seed for the random number generator.

    Attributes:
        seed (int): The seed for the random number generator.
    """

    def __init__(self, seed):
        super().__init__(seed, method="log_loss")


class MonteCarloEERMisclassification(MonteCarloEERStrategy):
    """MonteCarloEERMisclassification

    This class is used to sample instances from the pool of unlabeled instances based on the Monte Carlo EER method with
    the misclassification loss method.

    Args:
        seed (int): The seed for the random number generator.

    Attributes:
        seed (int): The seed for the random number generator.
    """

    def __init__(self, seed):
        super().__init__(seed, method="misclassification_loss")


class DiscriminativeQueryStrategy(ActiveMLQueryStrategy):
    def __init__(self, seed):
        super().__init__(seed=seed, qs=DiscriminativeAL(random_state=seed))

    def sample(self, learner, X_l, y_l, X_u, num_queries):
        nan_labels = np.full(len(X_u), np.nan, dtype=float)
        queried_ids = self.qs.query(
            X=np.concatenate([X_l, X_u]),
            y=np.concatenate([y_l, nan_labels]),
            discriminator=SklearnClassifier(learner),
            batch_size=num_queries,
        )
        queried_original_ids = queried_ids - len(y_l)
        return queried_original_ids


class ActiveMLEnsembleQueryStrategy(ActiveMLQueryStrategy):
    """ActiveMLEnsembleQueryStrategy

    This class is an abstract class for active learning query strategies that are ensemble-based. The query strategies
    are used to sample instances from the pool of unlabeled instances.

    Args:
        seed (int): The seed for the random number generator.
        qs: object
        ensemble_size (int): The size of the ensemble.

    Attributes:
        seed (int): The seed for the random number generator.
        qs: object
        ensemble_size (int): The size of the ensemble.
    """

    def __init__(self, seed, qs, ensemble_size):
        super().__init__(seed=seed, qs=qs)
        self.ensemble_size = ensemble_size

    def sample(self, learner, X_l, y_l, X_u, num_queries):
        learners = [SklearnClassifier(learner)] * self.ensemble_size
        nan_labels = np.full(len(X_u), np.nan, dtype=float)
        queried_ids = self.qs.query(
            X=np.concatenate([X_l, X_u]),
            y=np.concatenate([y_l, nan_labels]),
            ensemble=learners,
            fit_ensemble=True,
            batch_size=num_queries,
        )
        queried_original_ids = queried_ids - len(y_l)
        return queried_original_ids

    def get_params(self):
        params = super().get_params()
        params["ensemble_size"] = self.ensemble_size
        return params


class QueryByCommitteeQueryStrategy(ActiveMLEnsembleQueryStrategy):
    """QueryByCommitteeQueryStrategy

    This class is used to sample instances from the pool of unlabeled instances based on the Query by Committee method.

    Args:
        seed (int): The seed for the random number generator.
        method (str): The method for Query by Committee.
        ensemble_size (int): The size of the ensemble.

    Attributes:
        seed (int): The seed for the random number generator.
        qs: object
        ensemble_size (int): The size of the ensemble.
    """

    def __init__(self, seed, method, ensemble_size):
        super().__init__(seed=seed, qs=QueryByCommittee(method=method, random_state=seed), ensemble_size=ensemble_size)


class EnsemblePseudoRandomizedQueryStrategy(PseudoRandomizedQueryStrategy):
    def __init__(self, seed, ensemble_size):
        super().__init__(seed=seed)
        self.ensemble_size = ensemble_size

    @abstractmethod
    def sample(self, learner, X_l, y_l, X_u, num_queries):
        pass

    def get_params(self):
        return {"seed": self.seed, "ensemble_size": self.ensemble_size}


class QueryByCommitteeEntropyQueryStrategy(QueryByCommitteeQueryStrategy):
    """QueryByCommitteeEntropyQueryStrategy

    This class is used to sample instances from the pool of unlabeled instances based on the Query by Committee method
    with entropy as measure of ensemble disagreement.

    Args:
        seed (int): The seed for the random number generator.
        ensemble_size (int): The size of the ensemble.

    Attributes:
        seed (int): The seed for the random number generator.
        ensemble_size (int): The size of the ensemble.
    """

    def __init__(self, seed, ensemble_size):
        super().__init__(seed=seed, method="vote_entropy", ensemble_size=ensemble_size)


class QueryByCommitteeKLQueryStrategy(QueryByCommitteeQueryStrategy):
    """QueryByCommitteeKLQueryStrategy

    This class is used to sample instances from the pool of unlabeled instances based on the Query by Committee method
    with KL-divergence as measure of ensemble disagreement.

    Args:
        seed (int): The seed for the random number generator.
        ensemble_size (int): The size of the ensemble.

    Attributes:
        seed (int): The seed for the random number generator.
        ensemble_size (int): The size of the ensemble.
    """

    def __init__(self, seed, ensemble_size):
        super().__init__(seed=seed, method="KL_divergence", ensemble_size=ensemble_size)


class BatchBaldQueryStrategy(ActiveMLEnsembleQueryStrategy):
    """BatchBaldQueryStrategy

    This class is used to sample instances from the pool of unlabeled instances based on the BatchBALD method.

    Args:
        seed (int): The seed for the random number generator.
        ensemble_size (int): The size of the ensemble.

    Attributes:
        seed (int): The seed for the random number generator.
        ensemble_size (int): The size of the ensemble.
    """

    def __init__(self, seed, ensemble_size):
        super().__init__(seed=seed, qs=BatchBALD(random_state=seed), ensemble_size=ensemble_size)


#########################
# self-implemented
#########################
class EmbeddingBasedQueryStrategy(PseudoRandomizedQueryStrategy):
    """EmbeddingBasedQueryStrategy

    This class is an abstract class for query strategies that are based on embeddings. The query strategies are used to
    sample instances from the pool of unlabeled instances.

    Args:
        seed (int): The seed for the random number generator.

    Attributes:
        seed (int): The seed for the random number generator.
    """

    def __init__(self, seed):
        super().__init__(seed=seed)

    @abstractmethod
    def sample(self, learner, X_l, y_l, X_u, num_queries):
        pass

    def compute_embedding(self, learner, X_l, y_l, X_u, transform_labeled=False):
        learner_fqn = fullname(learner)
        if learner_fqn == "tabpfn.scripts.transformer_prediction_interface.TabPFNClassifier":
            from alpbench.util.TorchUtil import TabPFNEmbedder

            clf = TabPFNEmbedder(X_l, y_l)
            if not transform_labeled:
                X_u = clf.forward(X_u, encode=True)
            else:
                X = np.concatenate((X_l, X_u))
                X_embeds = clf.forward(X, encode=True)
                X_l, X_u = X_embeds[: len(X_l)], X_embeds[len(X_l) :]
        elif learner_fqn == "pytorch_tabnet.tab_model.TabNetClassifier":
            from pytorch_tabnet.tab_model import TabNetClassifier

            clf = TabNetClassifier(verbose=0)
            from alpbench.util.TorchUtil import TimeLimitCallback

            clf.fit(X_l, y_l, callbacks=[TimeLimitCallback(180)])
            if not transform_labeled:
                X_u = clf.predict_proba(X_u, get_embeds=True)
            else:
                X = np.concatenate((X_l, X_u))
                X_embeds = clf.predict_proba(X, get_embeds=True)
                X_l, X_u = X_embeds[: len(X_l)], X_embeds[len(X_l) :]
        if not transform_labeled:
            return X_u
        else:
            return X_u, X_l


class TypicalClusterQueryStrategy(EmbeddingBasedQueryStrategy):
    """TypicalClusterQueryStrategy

    This class is used to sample instances from the pool of unlabeled instances based on the typicality of the instances
    in the clusters.

    Args:
        seed (int): The seed for the random number generator.

    Attributes:
        seed (int): The seed for the random number generator.
    """

    def __init__(self, seed):
        super().__init__(seed=seed)

    def sample(self, learner, X_l, y_l, X_u, num_queries):
        pool_size = len(y_l)
        num_cluster = pool_size + num_queries
        X_u, X_l = self.compute_embedding(learner, X_l=X_l, y_l=y_l, X_u=X_u, transform_labeled=True)
        kmeans = KMeans(n_clusters=num_cluster)
        X = np.concatenate((X_l, X_u))
        kmeans.fit(X)
        # We don't want labels from this group
        labeled_cluster_classes = np.unique(kmeans.labels_[:pool_size])
        # Array of the following type: position in array is the label, value is the size of the cluster
        cluster_sizes = [len(np.argwhere(kmeans.labels_ == i)) for i in range(num_cluster)]
        ids_by_size = np.argsort(-np.array(cluster_sizes))
        labels_of_sorted_clusters = np.arange(num_cluster)[ids_by_size]
        selected_ids = []
        # Create a mask for instances_ids to track selected elements
        mask = np.ones(len(kmeans.labels_), dtype=bool)
        # Iterate through the num_samples uncovered largest clusters
        for idx in ids_by_size:
            current_label = labels_of_sorted_clusters[idx]
            if current_label not in labeled_cluster_classes:
                # Get neighbors within this cluster
                instances_ids = np.argwhere(kmeans.labels_ == current_label).flatten()
                instances_ids = instances_ids[mask[instances_ids]]  # Apply the mask
                if len(instances_ids) == 0:
                    continue
                elif len(instances_ids) == 1:
                    selected_ids.append(instances_ids[0])
                    mask[instances_ids[0]] = False  # Update the mask
                    if len(selected_ids) == num_queries:
                        selected_ids = np.array(selected_ids).flatten()
                        return selected_ids - pool_size
                else:
                    instances = X[instances_ids]
                    # Compute typicality for each instance, append the one with highest typicality
                    typicalities = []
                    K = 20
                    for i, instance in enumerate(instances):
                        remaining_instances = np.delete(instances, i, axis=0)
                        dists = np.sqrt(np.sum((instance - remaining_instances) ** 2, axis=1))
                        if len(dists) < K:
                            dist = np.mean(dists)
                        else:
                            dist = np.mean(np.sort(dists)[:K])
                        typicality = 1 / (dist + 1e-8)
                        typicalities.append(typicality)
                    typicalities = np.array(typicalities)
                    selected_id = instances_ids[np.argmax(typicalities)]
                    selected_ids.append(selected_id)
                    mask[selected_id] = False  # Update the mask
                    if len(selected_ids) == num_queries:
                        selected_ids = np.array(selected_ids).flatten()
                        return selected_ids - pool_size


class LeastConfidentQueryStrategy(PseudoRandomizedQueryStrategy):
    """LeastConfidentQueryStrategy

    This class is used to sample instances from the pool of unlabeled instances based on least confidence.

    Args:
        seed (int): The seed for the random number generator.

    Attributes:
        seed (int): The seed for the random number generator.
    """

    def __init__(self, seed):
        super().__init__(seed=seed)

    def sample(self, learner, X_l, y_l, X_u, num_queries):
        probas = learner.predict_proba(X_u)
        least_confidence = 1 - np.max(probas, axis=1)
        least_confidence_ids = np.argsort(least_confidence)[-num_queries:]
        return least_confidence_ids


class EntropyQueryStrategy(PseudoRandomizedQueryStrategy):
    """EntropyQueryStrategy

    This class is used to sample instances from the pool of unlabeled instances based on entropy.

    Args:
        seed (int): The seed for the random number generator.

    Attributes:
        seed (int): The seed for the random number generator.
    """

    def __init__(self, seed):
        super().__init__(seed=seed)

    def sample(self, learner, X_l, y_l, X_u, num_queries):
        probas = learner.predict_proba(X_u)
        entropies = -np.sum(probas * np.log(probas + 1e-8), axis=1)
        entropy_ids = np.argsort(entropies)[-num_queries:]
        return entropy_ids


class MarginQueryStrategy(PseudoRandomizedQueryStrategy):
    """MarginQueryStrategy

    This class is used to sample instances from the pool of unlabeled instances based on the margin method.

    Args:
        seed (int): The seed for the random number generator.

    Attributes:
        seed (int): The seed for the random number generator.
    """

    def __init__(self, seed):
        super().__init__(seed=seed)

    def sample(self, learner, X_l, y_l, X_u, num_queries):
        probas = learner.predict_proba(X_u)
        sorted_probas = np.sort(probas, axis=-1)
        margins = sorted_probas[:, -1] - sorted_probas[:, -2]
        original_margin_ids = np.argsort(margins)[:num_queries]
        return original_margin_ids


class PowerMarginQueryStrategy(PseudoRandomizedQueryStrategy):
    """PowerMarginQueryStrategy

    This class is used to sample instances from the pool of unlabeled instances based on the power margin method.

    Args:
        seed (int): The seed for the random number generator.

    Attributes:
        seed (int): The seed for the random number generator.
    """

    def __init__(self, seed):
        super().__init__(seed=seed)

    def sample(self, learner, X_l, y_l, X_u, num_queries):
        probas = learner.predict_proba(X_u)
        sorted_probas = np.sort(probas, axis=-1)
        margins = sorted_probas[:, -1] - sorted_probas[:, -2]
        margins = 1 - np.array(margins)
        # power transform
        np.random.seed(self.seed)
        margins = np.log(margins + 1e-8) + np.random.gumbel(size=len(margins))
        original_margin_ids = np.argsort(margins)[-num_queries:]
        return original_margin_ids


class RandomMarginQueryStrategy(PseudoRandomizedQueryStrategy):
    """RandomMarginQueryStrategy

    This class is used to sample instances from the pool of unlabeled instances based on the random margin method.

    Args:
        seed (int): The seed for the random number generator.

    Attributes:
        seed (int): The seed for the random number generator.
    """

    def __init__(self, seed):
        super().__init__(seed=seed)

    def sample(self, learner, X_l, y_l, X_u, num_queries):
        num_for_margin = num_queries // 2
        num_for_random = num_queries - num_for_margin
        random_ids = RandomQueryStrategy(self.seed).sample(learner, X_l, y_l, X_u, num_for_random)
        if num_for_margin == 0:
            return random_ids
        else:
            margin_ids = MarginQueryStrategy(self.seed).sample(learner, X_l, y_l, X_u, num_for_margin)
            return np.concatenate((margin_ids, random_ids))


class MinMarginQueryStrategy(EnsemblePseudoRandomizedQueryStrategy):
    """MinMarginQueryStrategy

    This class is used to sample instances from the pool of unlabeled instances based on the minimum margin method.

    Args:
        seed (int): The seed for the random number generator.
        ensemble_size (int): The size of the ensemble.

    Attributes:
        seed (int): The seed for the random number generator.
        ensemble_size (int): The size of the ensemble.
    """

    def __init__(self, seed, ensemble_size):
        super().__init__(seed=seed, ensemble_size=ensemble_size)
        self.ensemble_size = ensemble_size

    def sample(self, learner, X_l, y_l, X_u, num_queries):
        clf = Ens(estimator=learner, num_estimators=self.ensemble_size, max_neighbors=len(y_l))
        clf.fit(X_l, y_l)
        all_probas = clf.predict_proba(X_u)
        margins = np.zeros((self.ensemble_size, all_probas.shape[0]))
        sorted_probas = np.sort(all_probas, axis=1)
        margins = sorted_probas[:, -1, :] - sorted_probas[:, -2, :]
        margins = np.min(margins, axis=-1)
        margin_ids = np.argsort(margins)[:num_queries]
        return margin_ids


class FalcunQueryStrategy(PseudoRandomizedQueryStrategy):
    """FalcunQueryStrategy

    This class is used to sample instances from the pool of unlabeled instances based on the FALCUN method.

    Args:
        seed (int): The seed for the random number generator.

    Attributes:
        seed (int): The seed for the random number generator.
    """

    def __init__(self, seed):
        super().__init__(seed=seed)

    def sample(self, learner, X_l, y_l, X_u, num_queries):
        probas = learner.predict_proba(X_u)
        sorted_probas = np.sort(probas, axis=-1)
        margins = sorted_probas[:, -1] - sorted_probas[:, -2]
        div_scores = 1 - margins
        gamma = 10
        selected_ids = []
        ids_to_choose_from = np.arange(len(margins))
        mask = np.ones(len(margins), dtype=bool)  # Initialize mask
        for round in range(num_queries):
            relevance = margins + div_scores
            relevance[np.isnan(relevance)] = 0
            prob = relevance[mask] ** gamma / np.sum(relevance[mask] ** gamma)
            # set nan to 0, afterwards normalize
            prob[np.isnan(prob)] = 0
            prob = prob / np.sum(prob)
            prob[np.isnan(prob)] = 0
            np.random.seed(self.seed)
            # if still a nan
            if np.sum(np.isnan(prob)) > 0:
                selected_id = np.random.choice(ids_to_choose_from[mask])
            else:
                selected_id = np.random.choice(ids_to_choose_from[mask], p=prob)
            selected_ids.append(selected_id)
            # update div scores
            x_q = probas[selected_id]
            mask[selected_id] = False
            # remove selected id
            distances = np.sum(abs(probas - x_q), axis=1)
            div_scores = np.minimum(div_scores, distances)
            # normalize
            div_scores = (div_scores - np.min(div_scores)) / (np.max(div_scores) - np.min(div_scores) + 1e-8)

        return selected_ids


class WeightedClusterQueryStrategy(EmbeddingBasedQueryStrategy):
    """WeightedClusterQueryStrategy

    This class is used to sample instances from the pool of unlabeled instances based on the weighted cluster method.

    Args:
        seed (int): The seed for the random number generator.

    Attributes:
        seed (int): The seed for the random number generator.
    """

    def __init__(self, seed):
        super().__init__(seed=seed)

    def sample(self, learner, X_l, y_l, X_u, num_queries):
        scores = learner.predict_proba(X_u) + 1e-8
        entropy = -np.sum(scores * np.log(scores), axis=1)
        num_classes = num_queries
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

            per_class = num_queries // num_classes + 1
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

            return selected_ids[0:num_queries]


class KMeansQueryStrategy(EmbeddingBasedQueryStrategy):
    """KMeansQueryStrategy

    This class is used to sample instances from the pool of unlabeled instances based on the KMeans method.

    Args:
        seed (int): The seed for the random number generator.

    Attributes:
        seed (int): The seed for the random number generator.
    """

    def __init__(self, seed):
        super().__init__(seed=seed)

    def sample(self, learner, X_l, y_l, X_u, num_queries):
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

            per_class = num_queries // num_classes + 1
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

            return selected_ids[0:num_queries]


class ClusterMarginQueryStrategy(EmbeddingBasedQueryStrategy):
    """ClusterMargin

    This class is used to sample instances from the pool of unlabeled instances based on the cluster margin method.

    Args:
        seed (int): The seed for the random number generator.

    Attributes:
        seed (int): The seed for the random number generator.
    """

    def __init__(self, seed):
        super().__init__(seed=seed)

    def sample(self, learner, X_l, y_l, X_u, num_queries):
        m = 10
        probas = learner.predict_proba(X_u)
        X_u = self.compute_embedding(learner, X_l, y_l, X_u)
        num_clusters = min((len(X_l) + len(X_u)) // m, len(X_u))
        clustering = AgglomerativeClustering(n_clusters=num_clusters).fit(X_u)

        sorted_probas = np.sort(probas, axis=-1)
        margins = sorted_probas[:, -1] - sorted_probas[:, -2]

        # retrieve m*num_queries instances with smallest margins
        num_to_retrieve = m * num_queries
        ids_to_retrieve = np.argsort(margins)[:num_to_retrieve]
        # from those sample diversity-based through clustering
        cluster_belongings = clustering.labels_[ids_to_retrieve]
        # sort cluster by size
        cluster_dict = {}
        for i in set(cluster_belongings):
            cluster_dict[i] = len(np.argwhere(cluster_belongings == i))
        # sort by size
        keys = np.array(list(cluster_dict.keys()))
        values = np.array(list(cluster_dict.values()))
        sorted_indices = np.argsort(values)
        sorted_keys = keys[sorted_indices]

        selected_ids = []
        mask = np.ones(len(cluster_belongings), dtype=bool)

        while len(selected_ids) < num_queries:
            for key in sorted_keys:
                to_sample_from = np.argwhere((cluster_belongings == key) & mask).flatten()
                if len(to_sample_from) == 0:
                    continue
                else:
                    np.random.seed(self.seed)
                    id = np.random.choice(to_sample_from)
                    selected_ids.append(ids_to_retrieve[id])
                    mask[id] = False
                    if len(selected_ids) == num_queries:
                        return selected_ids

        return selected_ids[:num_queries]


class MaxEntropyQueryStrategy(EnsemblePseudoRandomizedQueryStrategy):
    """MaxEntropyQueryStrategy

    This class is used to sample instances from the pool of unlabeled instances based on the maximum entropy method.

    Args:
        seed (int): The seed for the random number generator.
        ensemble_size (int): The size of the ensemble.

    Attributes:
        seed (int): The seed for the random number generator.
        ensemble_size (int): The size of the ensemble.
    """

    def __init__(self, seed, ensemble_size):
        super().__init__(seed=seed, ensemble_size=25)
        self.ensemble_size = ensemble_size

    def sample(self, learner, X_l, y_l, X_u, num_queries):
        clf = Ens(estimator=learner, num_estimators=self.ensemble_size, max_neighbors=len(y_l))
        clf.fit(X_l, y_l)
        probas = clf.predict_proba(X_u).mean(axis=-1)
        entropies = -np.sum(probas * np.log(probas + 1e-8), axis=1)
        entropy_ids = np.argsort(entropies)[-num_queries:]
        return entropy_ids


class CoreSetQueryStrategy(EmbeddingBasedQueryStrategy):
    """CoreSetQueryStrategy

    This class is used to sample instances from the pool of unlabeled instances based on the core set method.

    Args:
        seed (int): The seed for the random number generator.

    Attributes:
        seed (int): The seed for the random number generator.
    """

    def __init__(self, seed):
        super().__init__(seed=seed)

    def sample(self, learner, X_l, y_l, X_u, num_queries):
        X_u, X_l = self.compute_embedding(learner, X_l, y_l, X_u, transform_labeled=True)
        selected_ids = []
        # Initialize the mask to all True, meaning all samples are initially available
        mask = np.ones(X_u.shape[0], dtype=bool)
        for round in range(num_queries):
            active_set = X_l[:, np.newaxis, :]
            inactive_set = X_u[np.newaxis, :, :]

            # Apply the mask to inactive_set
            inactive_set_masked = inactive_set[:, mask, :]

            distances = np.linalg.norm(active_set - inactive_set_masked, axis=2)

            # compute distance to closest neighbor
            dists = distances.min(axis=0)
            # get the id of the instance with the highest distance
            selected_id = np.argmax(dists)
            original_selected_id = np.where(mask)[0][selected_id]

            selected_ids.append(original_selected_id)

            # remove selected id from the mask
            mask[original_selected_id] = False

            # add the selected instance to X_l
            X_l = np.concatenate((X_l, X_u[original_selected_id].reshape(1, -1)))

        return selected_ids


class BALDQueryStrategy(EnsemblePseudoRandomizedQueryStrategy):
    """BALDQueryStrategy

    This class is used to sample instances from the pool of unlabeled instances based on the BALD method.

    Args:
        seed (int): The seed for the random number generator.
        ensemble_size (int): The size of the ensemble.

    Attributes:
        seed (int): The seed for the random number generator.
        ensemble_size (int): The size of the ensemble.
    """

    def __init__(self, seed, ensemble_size):
        super().__init__(seed=seed, ensemble_size=25)
        self.ensemble_size = ensemble_size

    def sample(self, learner, X_l, y_l, X_u, num_queries):
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
        mutual_info_ids = np.argsort(mutual_info)[-num_queries:]
        return mutual_info_ids


class PowerBALDQueryStrategy(EnsemblePseudoRandomizedQueryStrategy):
    """PowerBALDQueryStrategy

    This class is used to sample instances from the pool of unlabeled instances based on a power version of BALD.

    Args:
        seed (int): The seed for the random number generator.
        ensemble_size (int): The size of the ensemble.

    Attributes:
        seed (int): The seed for the random number generator.
        ensemble_size (int): The size of the ensemble.
    """

    def __init__(self, seed, ensemble_size):
        super().__init__(seed=seed, ensemble_size=25)
        self.ensemble_size = ensemble_size

    def sample(self, learner, X_l, y_l, X_u, num_queries):
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
        mutual_info_ids = np.argsort(mutual_info)[-num_queries:]
        return mutual_info_ids


class QBCVarianceRatioQueryStrategy(EnsemblePseudoRandomizedQueryStrategy):
    """QBCVarianceRatioQueryStrategy

    This class is used to sample instances from the pool of unlabeled instances based on QBC method with variance
    ratio as measure of ensemble disagreement.

    Args:
        seed (int): The seed for the random number generator.
        ensemble_size (int): The size of the ensemble.

    Attributes:
        seed (int): The seed for the random number generator.
        ensemble_size (int): The size of the ensemble.
    """

    def __init__(self, seed, ensemble_size):
        super().__init__(seed=seed, ensemble_size=25)
        self.ensemble_size = ensemble_size

    def sample(self, learner, X_l, y_l, X_u, num_queries):
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
        vr_ids = np.argsort(variance_ratios)[-num_queries:]
        return vr_ids

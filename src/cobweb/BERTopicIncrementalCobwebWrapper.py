"""
Wrapper to make CobwebClusterer class ready for BERTopic Library!!

https://maartengr.github.io/BERTopic/getting_started/online/online.html

Need to understand how we can gradually update the prior variance (ASSUMING IT DOES MATTER)
BUT it looks like prior_var doesn't matter so we can set it to some default!
"""

import torch
import numpy as np

from src.cobweb.CobwebClusterer import CobwebClusterer

class BERTopicIncrementalCobwebWrapper:

    def __init__(self, max_clusters=100, min_cluster_size=5, leaf_ratio=0.2):
        """
        Initialization of a Clustering Wrapper that should slot right into BERTopic
        
        :param min_cluster_size: Minimum number of nodes per cluster!
        :param leaf_ratio: Maximum number of leaf-to-node level ratio! For auto-selecting the deepest point at which we can form clusters.
        
        FORCES an adaptive implementation of clustering!
        """
        self.max_clusters = max_clusters
        self.min_cluster_size = min_cluster_size
        self.leaf_ratio = leaf_ratio
        self.cobweb = None

    def partial_fit(self, X):
        """
        We're going to allow this script to experimentally calculate and save the prior variance,
        initialize the Cobweb structure, and return a set of partial labels
        """
        buffer_texts = ["NOTEXT" for _ in range(len(X))]

        if not self.cobweb:
            self.cobweb = CobwebClusterer(
                transition_depth=-1,
                prior_var=None,
                corpus=buffer_texts,
                corpus_embeddings=X
            )
        else:
            self.cobweb.add_sentences(buffer_texts, X)

        _, level_counts, leaf_counts, _ = self.cobweb.tree.analyze_structure(verbose=True)

        empirical_transition_depth = 0

        for level in level_counts.keys():
            if (level_counts[level] - leaf_counts.get(level, 0) < self.max_clusters) and (leaf_counts.get(level, 0) / level_counts[level] <= self.leaf_ratio): # setting this to be a ratio but maybe we do a flat number of leaves?
                empirical_transition_depth = level
            else:
                break

        new_labels = self.cobweb._gather_clusters(
            high_count_thres=self.min_cluster_size,
            transition_depth=empirical_transition_depth
        ).cpu()

        self.labels_ = new_labels[-len(X):]

        return self

    def predict(self, X):
        labels, scores = self.cobweb.predict_clusters(X)
        labels = labels.cpu()
        scores = scores.cpu()
        return labels


class BERTopicPersistentCobwebWrapper:
    """Stateful Cobweb wrapper with fit=partial_fit and reusable tree state.

    Accepts an optional pre-existing CobwebClusterer to continue training. Stores
    full labels_ (all docs) after each update so callers can persist topics.
    """

    def __init__(self, cobweb: CobwebClusterer | None = None, max_clusters: int = 100, min_cluster_size: int = 5, leaf_ratio: float = 0.2):
        self.max_clusters = max_clusters
        self.min_cluster_size = min_cluster_size
        self.leaf_ratio = leaf_ratio
        self.cobweb = cobweb
        self.labels_ = None

    def partial_fit(self, X):
        buffer_texts = ["NOTEXT" for _ in range(len(X))]

        if self.cobweb is None:
            self.cobweb = CobwebClusterer(
                transition_depth=-1,
                prior_var=None,
                corpus=buffer_texts,
                corpus_embeddings=X,
            )
        else:
            self.cobweb.add_sentences(buffer_texts, X)

        _, level_counts, leaf_counts, _ = self.cobweb.tree.analyze_structure(verbose=True)

        empirical_transition_depth = 0
        for level in level_counts.keys():
            if (level_counts[level] - leaf_counts.get(level, 0) < self.max_clusters) and (leaf_counts.get(level, 0) / level_counts[level] <= self.leaf_ratio):
                empirical_transition_depth = level
            else:
                break

        self.labels_ = self.cobweb._gather_clusters(
            high_count_thres=self.min_cluster_size,
            transition_depth=empirical_transition_depth,
        ).cpu()
        
        return self

    def fit(self, X):
        return self.partial_fit(X)

    def predict(self, X):
        labels, scores = self.cobweb.predict_clusters(X)
        return labels.cpu()

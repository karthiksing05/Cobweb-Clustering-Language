"""
Wrapper to make CobwebClusterer class ready for BERTopic Library!!

BERTopic needs the below structure to function appropriately:
class ClusterModel:
    def fit(self, X):
        self.labels_ = None
        return self

    def predict(self, X):
        return X

So we merely need to create a labels list when we fit the predictor (and remember we can create a -1 label
that works for any leaves at the level we choose).
"""

from src.cobweb.CobwebClusterer import CobwebClusterer

class BERTopicCobwebWrapper:

    def __init__(self, cluster_level=4, min_cluster_size=5):
        """
        Initialization of a Clustering Wrapper that should slot right into BERTopic
        
        :param cluster_level: Level from which to parse clusters in the Cobweb hierarchy 
                              (soon to be extended to hierarchical topic modeling!)
        :param min_cluster_size: Description
        """
        self.cluster_level = cluster_level
        self.min_cluster_size = min_cluster_size

    def fit(self, X):
        """
        We're going to allow this script to experimentally calculate and save the prior variance,
        initialize the Cobweb structure, save all labels.
        """
        buffer_texts = ["NOTEXT" for i in range(len(X))]

        self.cobweb = CobwebClusterer(
            transition_depth=self.cluster_level,
            prior_var=None,
            corpus=buffer_texts,
            corpus_embeddings=X
        )

        self.labels_ = self.cobweb._gather_clusters(high_count_thres=self.min_cluster_size).cpu()

        print(self.labels_)

        self.cobweb.tree.analyze_structure()

        return self

    def predict(self, X):
        labels, scores = self.cobweb.predict_clusters(X)
        labels = labels.cpu()
        scores = scores.cpu()
        return labels

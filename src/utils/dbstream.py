"""
Helper folder to generate a DBSTREAM clustering model for Incremental Topic Modeling
"""

from river import stream
from river import cluster

class DBSTREAMRiver:
    def __init__(self):
        self.model = cluster.DBSTREAM()

    def partial_fit(self, umap_embeddings):
        for umap_embedding, _ in stream.iter_array(umap_embeddings):
            self.model.learn_one(umap_embedding)

        labels = []
        for umap_embedding, _ in stream.iter_array(umap_embeddings):
            label = self.model.predict_one(umap_embedding)
            labels.append(label)

        self.labels_ = labels
        return self
"""
Incremental wrapper around a BERTopic model.

For non-persistent clusterers, each call to partial_fit refits the BERTopic
model on the full corpus accumulated so far. When the wrapped clusterer is
stateful (e.g., BERTopicPersistentCobwebWrapper), set use_persistent_clusterer
so only the clusterer ingests the new batch while topic representations are
recomputed on the full corpus.
"""

from __future__ import annotations

import logging
from typing import List, Sequence

import numpy as np
from bertopic import BERTopic

logger = logging.getLogger(__name__)


class BERTopicRefitWrapper:
    """Refit-or-append wrapper for BERTopic.

    Maintains a running corpus. On ``partial_fit`` it either refits the wrapped
    BERTopic model on all seen documents (default) or, when configured for a
    persistent clusterer, only updates that clusterer with the incoming batch
    and refreshes topic representations on the full corpus.
    """

    def __init__(self, base_model: BERTopic, use_persistent_clusterer: bool = False):
        self.base_model = base_model
        self.use_persistent_clusterer = use_persistent_clusterer
        self._docs: List[str] = []
        self._fitted = False

    def partial_fit(self, batch_docs: Sequence[str]):
        if not batch_docs:
            return self

        clean_docs = [doc for doc in batch_docs if str(doc).strip()]
        if not clean_docs:
            return self

        self._docs.extend(clean_docs)

        if self.use_persistent_clusterer:
            return self._partial_fit_persistent(clean_docs)

        return self._refit_full_corpus()

    def fit(self, docs: Sequence[str]):
        return self.partial_fit(docs)

    def _refit_full_corpus(self):
        self.base_model.fit(self._docs)
        self._fitted = True
        if hasattr(self.base_model, "hdbscan_model") and hasattr(self.base_model.hdbscan_model, "labels_"):
            self.base_model.labels_ = np.asarray(self.base_model.hdbscan_model.labels_)
        return self

    def _partial_fit_persistent(self, batch_docs: Sequence[str]):
        clusterer = getattr(self.base_model, "hdbscan_model", None)
        if clusterer is None or not hasattr(clusterer, "partial_fit"):
            logger.warning(
                "Persistent clusterer path requested but clusterer lacks partial_fit; "
                "falling back to full refit"
            )
            return self._refit_full_corpus()

        try:
            embeddings = self._embed_documents(batch_docs)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Falling back to full refit due to embedding failure: %s", exc)
            return self._refit_full_corpus()

        if not self._fitted:
            # First-time fit to initialize BERTopic internals (ctfidf, topic_embeddings, etc.).
            self.base_model.fit(self._docs, embeddings=embeddings)
            self._fitted = True
            self.base_model.labels_ = np.asarray(self.base_model.hdbscan_model.labels_)
            return self

        clusterer.partial_fit(embeddings)

        labels = getattr(clusterer, "labels_", None)
        if labels is None:
            logger.warning("Clusterer did not expose labels_; falling back to full refit")
            return self._refit_full_corpus()

        labels = np.asarray(labels)
        if labels.shape[0] != len(self._docs):
            logger.warning(
                "Label length %d does not match corpus size %d; falling back to full refit",
                labels.shape[0],
                len(self._docs),
            )
            return self._refit_full_corpus()

        self.base_model.labels_ = labels

        # Recompute topic representations over the full corpus using the
        # updated labels. BERTopic.update_topics returns topics; depending on
        # the installed version, it may also return a tuple (topics, probs).
        updated_topics = self.base_model.update_topics(self._docs, topics=labels)
        if updated_topics is not None:
            try:
                topics_only = updated_topics[0] if isinstance(updated_topics, tuple) else updated_topics
                self.base_model.topics_ = topics_only
            except Exception:  # noqa: BLE001
                logger.debug("Unable to set topics_ from update_topics return")

        return self

    def get_topics(self):
        return self.base_model.get_topics()

    def transform(self, docs: Sequence[str]):
        return self.base_model.transform(docs)

    def _embed_documents(self, docs: Sequence[str]):
        extractor = getattr(self.base_model, "_extract_embeddings", None)
        if callable(extractor):
            try:
                extracted = extractor(docs, method="document")
                if isinstance(extracted, tuple):
                    for item in extracted:
                        if hasattr(item, "shape"):
                            return item
                return extracted
            except Exception as exc:  # noqa: BLE001
                logger.debug("Embedding via _extract_embeddings failed: %s", exc)

        emb_model = getattr(self.base_model, "embedding_model", None) or getattr(self.base_model, "_embedding_model", None)
        if emb_model is None:
            raise ValueError("No embedding model available for BERTopicRefitWrapper")

        if hasattr(emb_model, "embed_documents"):
            return emb_model.embed_documents(docs)
        if hasattr(emb_model, "encode"):
            return emb_model.encode(docs, show_progress_bar=False)
        if callable(emb_model):
            return emb_model(docs)

        raise ValueError("Unsupported embedding model in BERTopicRefitWrapper")

    def __getattr__(self, name):
        return getattr(self.base_model, name)

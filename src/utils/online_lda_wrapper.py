"""
Lightweight wrapper exposing a BERTopic-like interface for online LDA (gensim).

Provides partial_fit/transform/get_topics so it can plug into the incremental
benchmark runner alongside BERTopic models.
"""

from __future__ import annotations

from typing import List, Tuple, Dict, Optional

import numpy as np
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel
from sklearn.feature_extraction.text import CountVectorizer


class OnlineLDAWrapper:
    """Online LDA with a BERTopic-like API (partial_fit/transform/get_topics)."""

    def __init__(
        self,
        num_topics: int = 20,
        alpha: str | float = "auto",
        eta: str | float = "auto",
        iterations: int = 50,
        random_state: int | None = 42,
    ) -> None:
        self.num_topics = num_topics
        self.alpha = alpha
        self.eta = eta
        self.iterations = iterations
        self.random_state = random_state

        self._analyzer = CountVectorizer(stop_words="english", token_pattern=r"(?u)\b\w\w+\b").build_analyzer()
        self.dictionary: Dictionary | None = None
        self.model: LdaModel | None = None

    def partial_fit(self, raw_documents: List[str]):
        tokens = [self._analyzer(doc) for doc in raw_documents]
        if self.dictionary is None:
            # Build the vocabulary once on the first batch; keep it fixed afterwards
            # to avoid gensim expanding internal matrices in-place and hitting index
            # alignment errors during log_perplexity/bound calculations.
            self.dictionary = Dictionary(tokens)
        corpus = [self.dictionary.doc2bow(toks) for toks in tokens]
        if not corpus:
            return self
        if self.model is None:
            self.model = LdaModel(
                corpus=corpus,
                id2word=self.dictionary,
                num_topics=self.num_topics,
                alpha=self.alpha,
                eta=self.eta,
                iterations=self.iterations,
                update_every=1,
                chunksize=len(corpus),
                random_state=self.random_state,
            )
        else:
            self.model.update(corpus)
        return self

    def fit(self, raw_documents: List[str]):
        return self.partial_fit(raw_documents)

    def transform(self, raw_documents: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        if self.model is None or self.dictionary is None:
            return np.array([]), np.array([])
        tokens = [self._analyzer(doc) for doc in raw_documents]
        corpus = [self.dictionary.doc2bow(toks) for toks in tokens]
        dists = [self.model.get_document_topics(doc, minimum_probability=0.0) for doc in corpus]
        # convert to dense matrix
        mat = np.zeros((len(dists), self.num_topics), dtype=float)
        for i, dist in enumerate(dists):
            for topic_id, prob in dist:
                mat[i, topic_id] = prob
        labels = np.argmax(mat, axis=1) if mat.size else np.array([])
        return labels, mat

    def get_topics(self, topn: int = 10) -> Dict[int, List[Tuple[str, float]]]:
        if self.model is None:
            return {}
        topics: Dict[int, List[Tuple[str, float]]] = {}
        for topic_id in range(self.num_topics):
            words = self.model.show_topic(topic_id, topn=topn)
            topics[topic_id] = words
        return topics

    @property
    def topic_embeddings_(self):
        # Not available for LDA; used to signal absence for similarity metrics.
        return None

    @property
    def c_tf_idf_(self):
        # Not available; keeps compatibility with runner checks.
        return None

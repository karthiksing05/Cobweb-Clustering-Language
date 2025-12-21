"""
Utility classes for running and evaluating BERTopic models.

The helpers below wrap dataset loading and multi-model evaluation so experiments
can stay concise in notebooks and scripts. Metrics mirror those commonly used in
the BERTopic ecosystem: c_v coherence, NPMI coherence, topic diversity, and
inter-topic similarity (cosine similarity between topic embeddings).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from bertopic import BERTopic
from gensim.corpora.dictionary import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


def _default_analyzer() -> callable:
	"""Return a default analyzer that mirrors CountVectorizer tokenization."""
	vectorizer = CountVectorizer(stop_words="english", token_pattern=r"(?u)\b\w\w+\b")
	return vectorizer.build_analyzer()

@dataclass
class BERTopicDataset:
	"""
	Lightweight dataset wrapper for BERTopic.

	It keeps pre-tokenized documents, a gensim dictionary, and a BoW corpus so
	downstream coherence computations are trivial. Documents are accepted as an
	iterable of strings; embeddings can optionally be precomputed and supplied.
	"""

	documents: List[str]
	tokens: List[List[str]]
	dictionary: Dictionary
	corpus: List[List[Tuple[int, int]]]
	embeddings: Optional[np.ndarray] = None

	@classmethod
	def from_texts(
		cls,
		documents: Iterable[str],
		embeddings: Optional[Sequence[Sequence[float]]] = None,
		analyzer: Optional[callable] = None,
	) -> "BERTopicDataset":
		docs = [doc.strip() for doc in documents if str(doc).strip()]
		if not docs:
			raise ValueError("No documents provided to BERTopicDataset")

		analyzer_fn = analyzer or _default_analyzer()
		tokens = [analyzer_fn(doc) for doc in docs]
		dictionary = Dictionary(tokens)
		corpus = [dictionary.doc2bow(doc_tokens) for doc_tokens in tokens]

		embed_array = None
		if embeddings is not None:
			embed_array = np.asarray(embeddings)
			if embed_array.shape[0] != len(docs):
				raise ValueError("Embeddings and documents must have the same length")

		return cls(documents=docs, tokens=tokens, dictionary=dictionary, corpus=corpus, embeddings=embed_array)

	@classmethod
	def from_text_file(
		cls,
		path: str | Path,
		encoding: str = "utf-8",
		analyzer: Optional[callable] = None,
	) -> "BERTopicDataset":
		text_path = Path(path)
		with text_path.open("r", encoding=encoding) as handle:
			docs = [line.strip() for line in handle if line.strip()]
		return cls.from_texts(docs, analyzer=analyzer)

	@classmethod
	def from_csv(
		cls,
		path: str | Path,
		text_column: str = "text",
		encoding: str = "utf-8",
		analyzer: Optional[callable] = None,
	) -> "BERTopicDataset":
		df = pd.read_csv(path, encoding=encoding)
		if text_column not in df.columns:
			raise ValueError(f"Column '{text_column}' not found in CSV {path}")
		docs = df[text_column].astype(str).tolist()
		embeddings = df["embedding"].tolist() if "embedding" in df.columns else None
		return cls.from_texts(docs, embeddings=embeddings, analyzer=analyzer)

	@property
	def size(self) -> int:
		return len(self.documents)


class BERTopicRunner:
	"""
	Run one or more BERTopic instances on a dataset and compute standard metrics.
	"""

	def __init__(self, topic_models: Sequence[BERTopic]):
		if not topic_models:
			raise ValueError("Provide at least one BERTopic instance to BERTopicRunner")
		self.topic_models = list(topic_models)

	def run(
		self,
		dataset: BERTopicDataset,
		top_n_words: int = 10,
	) -> List[Dict[str, float]]:
		"""
		Fit each model on the dataset and return metric dictionaries.

		Returns a list matching the input models order. Each entry contains the
		fitted model and the requested metrics.
		"""

		results: List[Dict[str, float]] = []
		for i, model in enumerate(self.topic_models):
			logger.info("Fitting BERTopic model %s on %d docs", model.hdbscan_model.__class__.__name__, dataset.size)
			model.fit(dataset.documents, embeddings=dataset.embeddings)
			metrics = self._compute_metrics(model, dataset, top_n_words=top_n_words)
			results.append({"model": model, **metrics})
		return results

	def _compute_metrics(
		self,
		model: BERTopic,
		dataset: BERTopicDataset,
		top_n_words: int,
	) -> Dict[str, float]:
		topic_words = self._extract_topic_words(model, top_n=top_n_words)
		coherence_cv = self._coherence(topic_words, dataset, measure="c_v")
		coherence_npmi = self._coherence(topic_words, dataset, measure="c_npmi")
		diversity = self._topic_diversity(topic_words)
		inter_sim = self._inter_topic_similarity(model)

		return {
			"coherence_c_v": coherence_cv,
			"coherence_npmi": coherence_npmi,
			"topic_diversity": diversity,
			"inter_topic_similarity": inter_sim,
		}

	@staticmethod
	def _extract_topic_words(model: BERTopic, top_n: int) -> List[List[str]]:
		topics = model.get_topics()
		if not topics:
			return []

		topic_words: List[List[str]] = []
		for topic_id, words in topics.items():
			if topic_id == -1:
				continue  # skip outlier topic
			topic_words.append([word for word, _ in words[:top_n]])
		return topic_words

	@staticmethod
	def _coherence(
		topic_words: List[List[str]],
		dataset: BERTopicDataset,
		measure: str,
	) -> float:
		if not topic_words:
			return float("nan")

		try:
			coherence_model = CoherenceModel(
				topics=topic_words,
				texts=dataset.tokens,
				corpus=dataset.corpus,
				dictionary=dataset.dictionary,
				coherence=measure,
			)
			return float(coherence_model.get_coherence())
		except ValueError:
			logger.warning("Unable to compute %s coherence; returning NaN", measure)
			return float("nan")

	@staticmethod
	def _topic_diversity(topic_words: List[List[str]]) -> float:
		if not topic_words:
			return float("nan")

		unique_terms = set()
		total = 0
		for words in topic_words:
			unique_terms.update(words)
			total += len(words)
		return float(len(unique_terms) / total) if total else float("nan")

	@staticmethod
	def _inter_topic_similarity(model: BERTopic) -> float:
		topics = model.get_topics()
		if not topics or len(topics) < 2:
			return float("nan")

		topic_order = list(topics.keys())
		indices = [i for i, tid in enumerate(topic_order) if tid != -1]

		def _to_dense(matrix):
			return matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)

		embeddings = getattr(model, "topic_embeddings_", None)
		if embeddings is not None:
			matrix = np.asarray(embeddings)
		elif getattr(model, "c_tf_idf_", None) is not None:
			matrix = _to_dense(model.c_tf_idf_)
		else:
			return float("nan")

		if matrix.shape[0] < 2 or not indices:
			return float("nan")

		filtered = matrix[indices]
		sim = cosine_similarity(filtered)
		upper = sim[np.triu_indices_from(sim, k=1)]
		return float(np.mean(upper)) if upper.size else float("nan")
